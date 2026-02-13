import json
import os
from pathlib import Path

import fire
import pandas as pd
import torch
import torchaudio
from muq import MuQMuLan
from torchaudio.transforms import Resample
from tqdm import tqdm

from editing.AudioEditingCode.evals.lpaps import LPAPS
from editing.AudioEditingCode.evals.meta_clap_consistency import CLAPTextConsistencyMetric
from editing.AudioEditingCode.evals.utils import calc_clap_win, calc_lpaps_win
from editing.edit_audios_flowedit_medley import prepare_dataset
from editing.AudioEditingCode.code.env import PATH_AUDIOS_MEDLEY, PATH_PROMPTS_MEDLEY, PATH_LOWER_BOUND_MEDLEY
from src.metrics.alignment import MusicAlignmentEval
DISABLE_TQDM = False



def prepare_data(path_edited_audio: str):
    df_musiccaps = prepare_dataset(Path(PATH_AUDIOS_MEDLEY), Path(PATH_PROMPTS_MEDLEY))
    target_prompts = []
    source_audios = []
    edits = []
    srs_src = []
    srs_edit = []
    classification_task = []
    for idx, row in df_musiccaps.iterrows():
        classification_task.append(row["edit_class"])
        target_prompts.append(row["editing_prompt"])
        source_audio, source_audio_sr = torchaudio.load(row["path_yt"])
        edit_audio, edit_audio_sr = torchaudio.load(path_edited_audio + f"/a{idx}.wav")
        # assert (
        #     source_audio_sr == edit_audio_sr
        # ), f"Track {row['path_yt']} (sr={source_audio_sr}) has different sample rate than edited audio a{idx}.wav (sr={edit_audio_sr})"
        # if edit_audio.shape[1] < source_audio.shape[1]:
        #     source_audio = source_audio[:, : edit_audio.shape[1]]
        source_audios.append(source_audio)
        edits.append(edit_audio)
        srs_src.append(source_audio_sr)
        srs_edit.append(edit_audio_sr)
    return target_prompts, source_audios, edits, srs_src, srs_edit, classification_task


def load_models(device):
    # calculate metrics - initialize models
    # LPAPS
    lpaps_model = LPAPS(
        net="clap",
        device=device,
        net_kwargs={
            "model_arch": "HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
            "chkpt": "music_audioset_epoch_15_esc_90.14.pt",
            "enable_fusion": "fusion" in "music_audioset_epoch_15_esc_90.14.pt",
        },
        checkpoint_path="res/clap/pretrained",
    )
    clap_model = CLAPTextConsistencyMetric(
        model_path=os.path.join("res/clap/pretrained", "music_audioset_epoch_15_esc_90.14.pt"),
        model_arch="HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
        enable_fusion="fusion" in "music_audioset_epoch_15_esc_90.14.pt",
    ).to(device)

    clap_model = clap_model.eval()

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    return lpaps_model, clap_model, mulan


def get_lpaps(source_audios, edits, srs_src, srs_edit, device):
    # load lpaps
    lpaps_model = LPAPS(
        net="clap",
        device=device,
        net_kwargs={
            "model_arch": "HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
            "chkpt": "music_audioset_epoch_15_esc_90.14.pt",
            "enable_fusion": "fusion" in "music_audioset_epoch_15_esc_90.14.pt",
        },
        checkpoint_path="res/clap/pretrained",
    )

    # process
    lpaps_source_target = {}
    with torch.no_grad():
        for audio_idx in tqdm(range(len(source_audios)), desc="Calculating LPAPS", disable=DISABLE_TQDM):
            lpaps_source_target[audio_idx] = calc_lpaps_win(
                lpaps_model=lpaps_model,
                aud1=source_audios[audio_idx],
                aud2=edits[audio_idx],
                sr1=srs_src[audio_idx],
                sr2=srs_edit[audio_idx],
                win_length=(10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None),
                overlap=0.1,
                method="mean",
                device=device,
            )
    lpaps_source_target_df = pd.DataFrame(list(lpaps_source_target.items()), columns=["audio_idx", "lpaps"])
    return lpaps_source_target_df


def get_clap(target_prompts, edits, srs_edit, device):
    clap_model = CLAPTextConsistencyMetric(
        model_path=os.path.join("res/clap/pretrained", "music_audioset_epoch_15_esc_90.14.pt"),
        model_arch="HTSAT-base" if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else "HTSAT-tiny",
        enable_fusion="fusion" in "music_audioset_epoch_15_esc_90.14.pt",
    ).to(device)
    clap_model = clap_model.eval()

    clap_target_targetp = {}
    with torch.no_grad():
        for audio_idx in tqdm(range(len(edits)), desc="Calculating CLAP", disable=DISABLE_TQDM):
            clap_target_targetp[audio_idx] = {
                "clap": calc_clap_win(
                    clap_model=clap_model,
                    aud=edits[audio_idx],
                    sr=srs_edit[audio_idx],
                    target_prompt=target_prompts[audio_idx],
                    win_length=10 if "fusion" not in "music_audioset_epoch_15_esc_90.14.pt" else None,
                    overlap=0.1,
                    method="mean",
                    device=device,
                ),
                "prompt": target_prompts[audio_idx],
            }
    clap_target_targetp_df = pd.DataFrame.from_dict(clap_target_targetp, orient="index")
    return clap_target_targetp_df


def get_mulan(target_prompts, edits, srs_edit, device, verbose=True):
    # init mulan

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    mulan_target_targetp = {}
    with torch.no_grad():
        all_similarities = []
        for audio_idx in tqdm(range(len(edits)), desc="Calculating MUQT", disable=(DISABLE_TQDM or not verbose)):
            all_texts = [target_prompts[audio_idx]]
            text_embeds = mulan(texts=all_texts)
            batch_audio = edits[audio_idx]
            batch_audio = Resample(srs_edit[audio_idx], 24000)(batch_audio)
            batch_wavs = batch_audio.squeeze(1).to(mulan.device)
            batch_embed = mulan(wavs=batch_wavs)
            batch_similarities = mulan.calc_similarity(batch_embed, text_embeds)
            all_similarities.append(batch_similarities.cpu())
        similarities = torch.cat(all_similarities, dim=0)

        per_prompt_sims = {}
        for audio_idx in range(len(edits)):
            per_prompt_sims[audio_idx] = {}
            p_idx = 0
            per_prompt_sims[audio_idx][f"muqt_sim_p{p_idx}"] = similarities[audio_idx, p_idx].item()
            per_prompt_sims[audio_idx][f"p{p_idx}"] = target_prompts[audio_idx]

    mulan_target_targetp_df = pd.DataFrame.from_dict(per_prompt_sims, orient="index")
    return mulan_target_targetp_df

def resample_audios(path_audio_orignal: Path, path_audio_resampled: Path, target_sr: int):
    for file in tqdm(path_audio_orignal.glob("*.wav"), desc=f"Resampling audios in {path_audio_orignal}"):
        audio, sr = torchaudio.load(file)
        audio = Resample(sr, target_sr)(audio)
        torchaudio.save(path_audio_resampled / file.name, audio, target_sr)

def calculate_source_distance_metrics(device: torch.device, path_edited_audio: str, path_lower_bound: str):
    path_edited_audio = Path(path_edited_audio).resolve()
    path_edited_audio_resampled = (path_edited_audio.parent / f"{path_edited_audio.name}_32k").resolve()
    if not path_edited_audio_resampled.exists():
        path_edited_audio_resampled.mkdir(parents=True, exist_ok=True)
        resample_audios(path_edited_audio, path_edited_audio_resampled, 32000)

    path_lower_bound = Path(path_lower_bound).resolve()
    path_lower_bound_resampled = (path_lower_bound.parent / f"{path_lower_bound.name}_32k").resolve()
    if not path_lower_bound_resampled.exists():
        path_lower_bound_resampled.mkdir(parents=True, exist_ok=True)
        resample_audios(path_lower_bound, path_lower_bound_resampled, 32000)

    evaluator = MusicAlignmentEval(sampling_rate=32000, device=device)
    metrics = evaluator.main(
        generate_files_path=str(path_edited_audio_resampled),
        groundtruth_path=str(path_lower_bound_resampled),
        limit_num=None,
    )
    return metrics


def main(path_audio: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_prompts, source_audios, edits, srs_src, srs_edit, classification_tasks = prepare_data(path_audio)
    path_save_metrics = Path(path_audio).parent
    lpaps_source_target_df = get_lpaps(source_audios, edits, srs_src, srs_edit, device)
    lpaps_source_target_df["classification_task"] = classification_tasks
    lpaps_source_target_df.to_csv((path_save_metrics / "lpaps_to_source.csv"))
    clap_target_targetp_df = get_clap(target_prompts, edits, srs_edit, device)
    clap_target_targetp_df["classification_task"] = classification_tasks
    clap_target_targetp_df.to_csv((path_save_metrics / "clap_to_target_prompt.csv"))
    mulan_target_targetp_df = get_mulan(target_prompts, edits, srs_edit, device)
    mulan_target_targetp_df["classification_task"] = classification_tasks
    mulan_target_targetp_df.to_csv((path_save_metrics / "mulan_to_target_prompt.csv"))

    final_results = {
        "LPAPS": {
            "mean": lpaps_source_target_df["lpaps"].mean(),
            "std": lpaps_source_target_df["lpaps"].std(),
        },
        "CLAP": {
            "mean": clap_target_targetp_df["clap"].mean(),
            "std": clap_target_targetp_df["clap"].std(),
        },
        "MUQT": {
            "mean": mulan_target_targetp_df["muqt_sim_p0"].mean(),
            "std": mulan_target_targetp_df["muqt_sim_p0"].std(),
        },
    }
    print(final_results)
    with open((path_save_metrics / "final_results.json"), "w") as f:
        json.dump(final_results, f)

    per_task_result = {
        task: {
            "LPAPS": {
                "mean": lpaps_source_target_df[lpaps_source_target_df["classification_task"] == task]["lpaps"].mean(),
                "std": lpaps_source_target_df[lpaps_source_target_df["classification_task"] == task]["lpaps"].std(),
            },
            "CLAP": {
                "mean": clap_target_targetp_df[clap_target_targetp_df["classification_task"] == task]["clap"].mean(),
                "std": clap_target_targetp_df[clap_target_targetp_df["classification_task"] == task]["clap"].std(),
            },
            "MUQT": {
                "mean": mulan_target_targetp_df[mulan_target_targetp_df["classification_task"] == task][
                    "muqt_sim_p0"
                ].mean(),
                "std": mulan_target_targetp_df[mulan_target_targetp_df["classification_task"] == task][
                    "muqt_sim_p0"
                ].std(),
            },
        }
        for task in list(set(classification_tasks))
    }
    with open((path_save_metrics / "per_task_results.json"), "w") as f:
        json.dump(per_task_result, f)

    source_distance_metrics = calculate_source_distance_metrics(device=device, path_edited_audio=path_audio, path_lower_bound=PATH_LOWER_BOUND_MEDLEY)

    with open((path_save_metrics / "source_distance_metrics.json"), "w") as f:
        json.dump(source_distance_metrics, f)


if __name__ == "__main__":
    fire.Fire(main)

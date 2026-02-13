#!/bin/bash

# "piano|a piano song"
#     "mood|a cheerful track"
#     "tempo|a fast track"
#     "female_vocals|This is a music of a female vocal singing"
#     "drums|a drums song"

# python steering/ace_steer/eval_steering_protocol.py --steering_dir steering/outputs/female_vocals/sae_diff/sequence_k125_pool_False_20260211231418 --eval_prompt "This is a music of a female vocal singing"

python steering/ace_steer/eval_steering_protocol.py --steering_dir steering/outputs/tempo/sae_tfidf/sequence_k123_pool_False_20260211231346 --eval_prompt "a fast track"

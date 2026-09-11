#!/bin/bash
# 一键回退：恢复 sampler 原文（其余改动均为 env 默认关闭，不设即原行为）
cd /home/y50063564/dspark_project/speculators
git checkout -- src/speculators/train/distributed_batch_sampler.py
echo "reverted distributed_batch_sampler.py to git HEAD version"
git diff --stat

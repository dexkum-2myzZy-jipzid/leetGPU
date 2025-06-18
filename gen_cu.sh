#!/bin/bash

# 检查参数
if [ $# -ne 1 ]; then
    echo "用法: $0 文件名"
    exit 1
fi

input="$1"

# 转小写，空格转下划线
filename=$(echo "$input" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

output="${filename}.cu"

touch "$output"
echo "$output 已生成"
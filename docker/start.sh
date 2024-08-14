#! /bin/bash
# ↑これはただのコメントではないので消してはいけない(#!/ bin/shではpermission denied, なしだとexec format errorになる)
#ssh serverを起動
# sudo su -
sudo /usr/sbin/sshd -D

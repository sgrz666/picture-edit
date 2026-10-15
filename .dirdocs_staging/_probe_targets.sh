set -u
# 应写清单：仓库 depth1 全量 + 结构层 depth2 + 实验/调研/数据层 depth<=3；排除编号叶子与第三方内部
cd /home/shangguanrz/project/pic-edit || exit 1
: > /tmp/target_repo.txt
for d in $(find . -maxdepth 1 -mindepth 1 -type d -not -path './.git*' -not -name '__pycache__' -not -name '.*' | sed 's|^\./||'); do
  echo "$d" >> /tmp/target_repo.txt
done
for top in src scripts tests configs docs reports patches third_party vram_lab inputs logs outputs caches data models; do
  find "$top" -mindepth 1 -maxdepth 2 -type d -not -name '__pycache__' -not -name '.*' 2>/dev/null >> /tmp/target_repo.txt
done
for top in experiments research results archive datasets add; do
  find "$top" -mindepth 1 -maxdepth 3 -type d 2>/dev/null \
    | grep -Ev '/[0-9]{2,8}(/|$)|/seq_[0-9]+(/|$)|/[0-9]{5}_[0-9]+(/|$)|/motion-[0-9]+(/|$)|__pycache__|/\.|/[0-9]{6,}(/|$)' >> /tmp/target_repo.txt
done
# add/tools 与 add/envs 的第三方内部不深入：只保留 depth<=2
awk -F/ '($1!="add/tools" && $1!="add/envs") || NF<=2' /tmp/target_repo.txt | sort -u > /tmp/target_repo_final.txt
echo "repo targets=$(wc -l < /tmp/target_repo_final.txt)"
cd /data/shangguanrz || exit 1
: > /tmp/target_data.txt
find . -maxdepth 1 -mindepth 1 -type d -not -name '.*' | sed 's|^\./||' >> /tmp/target_data.txt
for top in $(ls -d */ 2>/dev/null | tr -d '/'); do
  find "$top" -mindepth 1 -maxdepth 2 -type d -not -name '.*' 2>/dev/null >> /tmp/target_data.txt
done
awk -F/ 'NF<=4' /tmp/target_data.txt | grep -Ev '/[0-9]{2,8}$|/seq_[0-9]+$|/[0-9]{5}_[0-9]+$|__pycache__|/\.' | sort -u > /tmp/target_data_final.txt
echo "data targets=$(wc -l < /tmp/target_data_final.txt)"
echo '@@@ repo 样例'; head -8 /tmp/target_repo_final.txt
echo '@@@ data 样例'; head -8 /tmp/target_data_final.txt

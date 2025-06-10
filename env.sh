# if /scratch/doqihu is mounted, use it as home
if [ -d /scratch/doqihu ]; then
    home=/scratch/doqihu
else
    home=${:-$HOME}
fi
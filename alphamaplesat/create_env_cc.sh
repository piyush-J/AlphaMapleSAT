module load python/3.10
virtualenv --no-download ams_env
source ams_env/bin/activate
cd alphamaplesat
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt

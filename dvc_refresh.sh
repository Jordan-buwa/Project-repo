rm data/snapshots.dvc && git rm data/snapshots.dvc

dvc add data/snapshots

git add .

git commit -m "Fix: re-track data/snapshots after cache loss"

dvc push -r azureblob 

git push
ssh -l amcgover m3.massive.org.au
smux new-session --account=rl54
rsync -auv -e ssh /Users/amcg0011/Data/pia-tracking/cang_training amcgover@m3-dtn.massive.org.au:~/rl54/segmentation/training_data
rsync -auv -e ssh /Users/amcg0011/GitRepos/pia-tracking/MASSIVE  amcgover@m3-dtn.massive.org.au:~/rl54/segmentation/
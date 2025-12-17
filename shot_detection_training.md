# How to train a model to automatically classify shot-types for padel games.

We want to obtain pose data using MMPose from videos and feed a set number of frames so that a random forest can be trained on them.

use the .mmpose venv when running anything
## Important information

1 - I have a set of csvs at /home/daniele/shots_csvs/*.csv that contain information about shots being permformed on videos.
2 - The videos are at /home/daniele/videos
3 - Each .csv contains the type of shot, which frame started it and if it is the right or left player doing it.

## Summary of steps

1 - Extract poses of the correct player 15 frames before and after the frame number depicted in the csv.
2 - Create a clip for each shot with the overlayed pose to confirm its correctness.
3 - For each shot create a .csv such as {video_name}_{frame_from_csv}_{shot_type}_{left_or_right_player}.csv with the extracted pose keymarks.
4 - Take the pose of the person that is not doing the shot to have an "idle" class.
5 - Decide on clustering of all the classes to simplify at the beggining. 
6 - Use that information to train a RF classifier for shot detection.
# Extract Notes From Piano Music Videos
A Computer Vision Script to extract music notes from video.

This Script takes a video(Synthesia clips from youtube are recommended) and then extracts the piano from it and displays the key pressed by looking at the colors of each key.
This script is for people who want to practice a music piece but don't know how to read a music sheet.
This script when given an input video will produce three videos of different speeds for people with different skill level.
Those three videos will display the key pressed along with the piano on the top for assistance at each moment.
<hr>
<h2>Installation:</h2>

<ul>
<li>You need to have python3 and a few libraries:<ul>
<li>opencv</li>
<li>numpy</li>
<li>moviepy</li>
</ul></li>
<li>You need ffmpeg</li>
</ul>
<p>First create a folder Vids in the same dir as the script and download the videos in that folder</p>
<p>Open the script with an editor and just change the name of the song for the variable 'Video' and you can also choose the resolution you to load it in</p>
<p>Run the script and you will have a new folder called "Notes" and in it a folder with the name of the song which contains three video files:<ol><li>Normal - 1x</li><li>Slow - 0.5x</li><li>Intermediate - 0.8x</li></ol></p>
<p>Done!</p>

<hr>
<h2>Important Note:</h2>

The input Video must be a <strong>mp4</strong> file.

Let me be clear first, it's not <strong>PERFECT</strong>. I was able to load and extract notes from about 10 videos and all of them had similar color palette and overall look.

The clips I used were all taken from https://www.youtube.com/user/Marioverehrer2. This channel has many synthesia clips of many famous songs, and the code worked perfectly for most of his videos. When I tried clips from other channels which had different color palette and the size of the piano was different too, I started experiencing some bugs.

I would suggest to take clips from the channel mentioned above, but if you could'nt find the song of your choice then you may have to do some tweaking in the code.

You may change the range values to extract the piano in the initial frame.

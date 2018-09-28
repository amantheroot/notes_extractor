import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import time
import os
from shutil import copyfile
import subprocess
import sys

if not os.path.exists('Vids'):
	print("Please Create A New Folder \"Vids\" And Paste Your Clips There")
	sys.exit(0)

Video = "Fur Elise"# Choose
resolution = [640,360]# Choose

Video_path = "Vids/{}.mp4".format(Video)

clip = VideoFileClip(Video_path)
duration = clip.duration

cap = cv2.VideoCapture(Video_path)

one_time  = 0

SHEET = []

while True:
	_, frame = cap.read()

	if frame is None:
		break

	frame = cv2.resize(frame,(resolution[0],resolution[1]))
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	dim = frame.shape

	lower_red_white = np.array([0,0,200])
	upper_red_white = np.array([179,37,255])# May Need To Tweak

	mask = cv2.inRange(hsv, lower_red_white, upper_red_white)

	kernel = np.ones((20,3),np.uint8)

	opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
	closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

	# Piano And Notes Detection
	if one_time < 1:
		I = []
		for a in range(50):
			big_box_x = np.random.randint(dim[1])
			roi = closing[0:dim[0],big_box_x:big_box_x+1]
			i = 0
			for x in roi:
				value = x[0]
				if value == 255:
					break
				i+=1
			I.append(i)
		big_box_height = min(I)
		big_box_whites = max(I)

		if big_box_height == big_box_whites:
			one_time = 0
			continue

		roi_close = closing[big_box_height:dim[0],0:dim[1]]

		white_count_slice = int(np.round((big_box_whites-big_box_height)/2))
		roi_close_slice = roi_close[white_count_slice:white_count_slice+1,0:dim[1]]
		black_count = 0
		white_count = 0

		black_end_points = []
		fsb = False
		if roi_close_slice[0][0] == 0:
			started = True
			black_count += 1
			fsb = True
			black_end_points.append([0])
		else:
			started = False
			white_count  += 1
			fsb = False

		lw = 0
		wsc = 0
		white_spaces = []
		lsb = False
		for rcs in roi_close_slice[0]:
			if started:
				if rcs == 255:
					started = False
					white_count += 1
					wsc += 1
					black_end_points[black_count - 1].append(lw)
				elif lw == len(roi_close_slice[0]) - 1:
					lsb = True
					black_end_points[black_count - 1].append(lw)
			else:
				if rcs == 0:
					started = True
					black_count += 1
					white_spaces.append(wsc)
					wsc  = 0
					black_end_points.append([lw])
				else:
					if lw ==  len(roi_close_slice[0]) - 1:
						wsc += 1
						white_spaces.append(wsc)
						lsb = False
					else:
						wsc += 1
			lw += 1
		delta = 0.1
		wbyw1 = []
		KEY = []
		found_g = False
		num_g = 0
		while True:
			for w in range(len(white_spaces[0:-1])):
				fraction = white_spaces[w]/white_spaces[w+1]
				wbyw1.append(fraction)
			for w in wbyw1:
				if w > 1 - delta and w < 1 + delta:
					KEY.append("G")
					found_g = True
					num_g += 1
				else:
					KEY.append("Z")
			KEY.append("Z")
			if found_g:
				if num_g > int(np.round(white_count/5)) + 1:
					delta  = 0.8*delta
				else:
					break
			else:
				delta = 1.2*delta

		ref_keys_pos = ['G','A','BC','D','EF']
		ref_keys_neg = ['G','EF','D','BC','A']
		for  k in range(len(KEY)):
			if KEY[k] == "G":
				for rk  in range(len(ref_keys_pos)):
					times = 0
					while True:
						to_add = 5*times + rk
						if k + to_add >= len(KEY):
							break
						else:
							KEY[k+to_add] = ref_keys_pos[rk]
						times += 1
					times = 0
					while True:
						to_remove = 5*times + rk
						if k - to_remove < 0:
							break
						else:
							KEY[k-to_remove] = ref_keys_neg[rk]
						times += 1
		white_keys = []
		for key in KEY:
			list_key = list(key)
			if len(list_key) == 2:
				white_keys.append(list_key[0])
				white_keys.append(list_key[1])
			else:
				white_keys.append(key)

		white_count = len(white_keys)

		keys_with_sharp = ['C','D','F','G','A']
		piano_whites = ['C','D','E','F','G','A','B']
		all_keys = []
		for white in range(len(white_keys)):
			if white == 0 and fsb:
				hashed = piano_whites[piano_whites.index(white_keys[0])-1]
				hashed = hashed+"#"
				all_keys.append(hashed)
				all_keys.append(white_keys[white])
			elif white == len(white_keys)-1:
				if lsb:
					all_keys.append(white_keys[white])
					hashed = white_keys[white]+"#"
					all_keys.append(hashed)
				else:
					all_keys.append(white_keys[white])
			else:
				if white_keys[white] in keys_with_sharp:
					all_keys.append(white_keys[white])
					hashed = white_keys[white]+"#"
					all_keys.append(hashed)
				else:
					all_keys.append(white_keys[white])
		
		total_keys  =  len(all_keys)

		label =  0
		labled_keys = []
		for keys in range(total_keys):
			if all_keys[0] == 'C':
				if keys == 0:
					label = 0
				elif all_keys[keys] == 'C':
					label += 1
				string = all_keys[keys]+str(label)
				labled_keys.append(string)
			else:
				if all_keys[keys] == 'C':
					label += 1
				string = all_keys[keys]+str(label)
				labled_keys.append(string)

		print(labled_keys)

		print("No. Of Black Keys = ",black_count)
		print("No. Of White Keys = ",white_count)
		print("Total Keys = ",total_keys)

		all_white_keys = []
		all_black_keys = []
		for key in labled_keys:
			list_key = list(key)
			if not "#" in list_key:
				all_white_keys.append(key)
			else:
				all_black_keys.append(key)

		num_fws = 0
		num_lws = 0
		for key in labled_keys:
			list_key = list(key)
			if not "#" in list_key:
				num_fws += 1
			else:
				break
		for key in labled_keys:
			if key == all_black_keys[-1]:
				num_lws = len(labled_keys) - labled_keys.index(key) - 1

		###############################################################################################################################
		white_points = []

		quarter = 0.25
		half = 0.5
		quarter3 = 0.75


		if "C#" in all_black_keys[0] or "F#" in all_black_keys[0]:
			if num_fws == 1:
				white_points.append(black_end_points[0][0] - quarter*black_end_points[0][0])
			else:
				white_points.append(black_end_points[0][0] - quarter*black_end_points[0][0])
				white_points.append(black_end_points[0][0] - quarter3*black_end_points[0][0])
			white_points.append(black_end_points[0][1] + half*(black_end_points[1][0] - black_end_points[0][1]))
		elif "D#" in all_black_keys[0] or "A#" in all_black_keys[0]:
			white_points.append(black_end_points[0][0] - half*black_end_points[0][0])
			white_points.append(black_end_points[0][1] + quarter*(black_end_points[1][0] - black_end_points[0][1]))
		else:
			white_points.append(black_end_points[0][0] - half*black_end_points[0][0])
			white_points.append(black_end_points[0][1] + half*(black_end_points[1][0] - black_end_points[0][1]))


		if "C#" in all_black_keys[-1] or "F#" in all_black_keys[-1]:
			white_points.append(black_end_points[-1][0] - quarter*(black_end_points[-1][0] - black_end_points[black_count-2][1]))
			white_points.append(black_end_points[-1][1] + half*(dim[1] - black_end_points[-1][1]))
		elif "D#" in all_black_keys[-1] or "A#" in all_black_keys[black_count-1]:
			white_points.append(black_end_points[-1][0] - half*(black_end_points[-1][0] - black_end_points[black_count-2][1]))
			if num_lws ==  1:
				white_points.append(black_end_points[-1][1] + quarter*(dim[1] - black_end_points[-1][1]))
			else:
				white_points.append(black_end_points[-1][1] + quarter*(dim[1] - black_end_points[-1][1]))
				white_points.append(black_end_points[-1][1] + quarter3*(dim[1] - black_end_points[-1][1]))
		else:
			white_points.append(black_end_points[-1][0] - half*(black_end_points[-1][0] - black_end_points[black_count-2][1]))
			white_points.append(black_end_points[-1][1] + half*(dim[1] - black_end_points[-1][1]))


		for b in range(black_count):
			if b  == 0 or b == black_count-1:
				continue
			if "C#" in all_black_keys[b] or "F#" in all_black_keys[b]:
				white_points.append(black_end_points[b][0] - quarter*(black_end_points[b][0] - black_end_points[b-1][1]))
				white_points.append(black_end_points[b][1] + half*(black_end_points[b+1][0] - black_end_points[b][1]))
			elif "D#" in all_black_keys[b] or "A#" in all_black_keys[b]:
				white_points.append(black_end_points[b][0] - half*(black_end_points[b][0] - black_end_points[b-1][1]))
				white_points.append(black_end_points[b][1] + quarter*(black_end_points[b+1][0] - black_end_points[b][1]))
			else:
				white_points.append(black_end_points[b][0] - half*(black_end_points[b][0] - black_end_points[b-1][1]))
				white_points.append(black_end_points[b][1] + half*(black_end_points[b+1][0] - black_end_points[b][1]))
		

		white_points = list(np.round(np.sort(list(set(white_points)))).astype(np.int))

		#################################################################################################################################

		black_points = []
		for b in range(black_count):
			middle_point = black_end_points[b][0] + np.round((black_end_points[b][1] - black_end_points[b][0])/2).astype(np.int)
			black_points.append(middle_point)

		points = list(np.sort(white_points+black_points))

		repititions = 10
		heights = []
		for i in range(repititions):
			height = np.round((big_box_whites - big_box_height)*np.random.rand()).astype(np.int)
			heights.append(height)

		one_time += 1

	roi = frame[big_box_height:dim[0],0:dim[1]]
	roi_whiteblack = roi[0:big_box_whites-big_box_height,0:dim[1]]
	roi_close = closing[big_box_height:dim[0],0:dim[1]]
	roi_close_black = roi_close[0:big_box_whites-big_box_height,0:dim[1]]

	lower_red_black = np.array([0,0,0])
	upper_red_black = np.array([179,255,35])#choose ideal value for value

	mask_black = cv2.inRange(roi_whiteblack, lower_red_black, upper_red_black)

	kernel_black = np.ones((20,3),np.uint8)

	closing_black = cv2.morphologyEx(mask_black,cv2.MORPH_CLOSE,kernel_black)

	all_black = [True]*black_count
	for bp in black_points:
		for h in heights:
			if closing_black[h,bp] != 0:
				all_black[black_points.index(bp)] = False

	white_part  = big_box_whites - big_box_height + int(np.round((dim[0] - big_box_whites)/2))
	roi_white_slice  = roi_close[white_part:white_part+1,0:dim[1]]

	keys_pressed = []
	for i in range(len(roi_white_slice[0])):
		if (roi_white_slice[0][i] == 0 and i in white_points) or (i in black_points and all_black[black_points.index(i)] == True):
			keys_pressed.append(labled_keys[points.index(i)])

	SHEET.append(keys_pressed)

	cv2.imshow("Piano",frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

num_frames = len(SHEET)
fps = num_frames/duration
spf = round(1/fps, 7)

print(fps,spf)

NOTES = []
for note in SHEET:
	if len(note) == 0:
		string = ""
	elif len(note) > 1:
		note_tuple = tuple(note)
		string = "("
		for nstri in note_tuple:
			if nstri == note_tuple[-1]:
				string += nstri+")"
			else:
				string += nstri+","
	else:
		string = note[0]
	NOTES.append(string)
print(NOTES)

n = 0

note_dir = "Notes"
if not os.path.exists(note_dir):
	os.mkdir(note_dir)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
Notes_path = 'Notes/{} notes modded.mp4'.format(Video)

out = cv2.VideoWriter(Notes_path,fourcc, 20.0, (resolution[0],resolution[1]))
cap = cv2.VideoCapture(Video_path)

while True:
	if n >= num_frames:
		break

	text = NOTES[n]

	white = np.zeros((resolution[1],resolution[0],3),np.uint8)
	white[:] = (255,255,255)

	size = 1

	text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, size,2)[0]
	center_X = np.round((white.shape[1] - text_size[0])/2).astype(np.int64)
	center_Y = np.round((white.shape[0] - text_size[1])/2).astype(np.int64)

	cv2.putText(white, text, (center_X, center_Y),cv2.FONT_HERSHEY_TRIPLEX,size,(0,0,0),2,cv2.LINE_AA)

	_, frame = cap.read()
	frame = cv2.resize(frame, (resolution[0],resolution[1]))

	roi = frame[big_box_height:dim[0],0:dim[1]]
	roi_shape = roi.shape

	white[0:roi_shape[0],0:roi_shape[1]] = roi

	out.write(white)

	cv2.imshow("white",white)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	n+=1
	
	time.sleep(spf)

out.release()
cv2.destroyAllWindows()

note_clip = VideoFileClip(Notes_path)
note_duration = note_clip.duration

duration_fraction = duration/note_duration
print(duration_fraction)

Notes_out_path = 'Notes/{} notes vid.mp4'.format(Video)

return_value = subprocess.call([
    'ffmpeg',
    '-y',
    '-i', '{}'.format(Notes_path),
    '-vf', 'setpts={}*PTS'.format(duration_fraction),
    Notes_out_path,
])

if return_value:
    print("Failure")
else:
    print("Sucess!")

Audio_path = "Notes/{} notes audio.mp3".format(Video)

return_value = subprocess.call([
    'ffmpeg',
    '-y',
    '-i', '{}'.format(Video_path),
    '-f', 'mp3',
    '-ab', '192000',
    '-vn',
    Audio_path,
])

if return_value:
    print("Failure")
else:
    print("Sucess!")

Normal_notes_path = "Notes/{} notes.mp4".format(Video)
Slow_notes_path = "Notes/{} notes slow.mp4".format(Video)
Intermediate_notes_path = "Notes/{} notes intermediate.mp4".format(Video)

return_value = subprocess.call([
    'ffmpeg',
    '-y',
    '-i', '{}'.format(Notes_out_path),
    '-i', '{}'.format(Audio_path),
    '-map', '0:v',
    '-map', '1:a',
    '-c:v', 'copy',
    '-c:a', 'copy',
    '-shortest', Normal_notes_path,
])

if return_value:
    print("Failure")
else:
    print("Sucess!")

return_value = subprocess.call([
    'ffmpeg',
    '-y',
    '-i', '{}'.format(Normal_notes_path),
    '-filter_complex', '[0:v]setpts=2.0*PTS[v];[0:a]atempo=0.5[a]',
    '-map', '[v]',
    '-map', '[a]',
    Slow_notes_path,
])

if return_value:
    print("Failure")
else:
    print("Sucess!")

return_value = subprocess.call([
    'ffmpeg',
    '-y',
    '-i', '{}'.format(Normal_notes_path),
    '-filter_complex', '[0:v]setpts=1.25*PTS[v];[0:a]atempo=0.8[a]',
    '-map', '[v]',
    '-map', '[a]',
    Intermediate_notes_path,
])

if return_value:
    print("Failure")
else:
    print("Sucess!")

song_dir = "Notes/{}".format(Video)
if not os.path.exists(song_dir):
	os.mkdir(song_dir)

normal_copy_path = song_dir+"/{} notes.mp4".format(Video)
slow_copy_path = song_dir+"/{} notes slow.mp4".format(Video)
inter_copy_path = song_dir+"/{} notes intermediate.mp4".format(Video)

copyfile(Normal_notes_path,normal_copy_path)
copyfile(Slow_notes_path,slow_copy_path)
copyfile(Intermediate_notes_path,inter_copy_path)

os.remove(Notes_path)
os.remove(Audio_path)
os.remove(Notes_out_path)
os.remove(Normal_notes_path)
os.remove(Slow_notes_path)
os.remove(Intermediate_notes_path)

print("Done!")

print("- Made By Aman kumar")
print("https://github.com/amantheroot")
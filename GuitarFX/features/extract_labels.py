import os

def extract_label(file_path: str) -> str:
	"""
	Extract the effect label (single (not multi) label!) from
	the filename.

	Args:
		file_path (str): The filepath to the file.

	Returns:
		str: Effect label
	"""
	effect_labels = {
		'11': 'No Effect',
		'12': 'No Effect',
		'21': 'Feedback Delay',
		'22': 'Slapback Delay',
		'23': 'Reverb',
		'31': 'Chorus',
		'32': 'Flanger',
		'33': 'Phaser',
		'34': 'Tremolo',
		'35': 'Vibrato',
		'41': 'Distortion',
		'42': 'Overdrive'
	}

	file_name = os.path.basename(file_path)
	effect_code = file_name.split("-")[2][1:3]
	return effect_labels.get(effect_code, "Unknown Effect")
# Copyright (c) WaveVortex 
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parse the attributes of speaker and acoustic features.
"""

import numpy as np
from typing import Union, Tuple
from sparkvox.utils.data import *

GENDER_LABEL_MAP = {
    "f": "female", 
    "m": "male",
    "female": "female",
    "male": "male",
    }

AGE_LABEL_MAP = {
    "child": "Child",
    "teenager": "Teenager",
    "youth-adult": "Youth-Adult",
    "middle-aged": "Middle-aged",
    "elderly": "Elderly"
}

def age_group(age: int) -> str:
    """Get the age group of a person."""
    if age < 12:
        return "Child"
    elif 12 <= age < 18:
        return "Teenager"
    elif 18 <= age < 40:
        return "Youth-Adult"
    elif 40 <= age < 60:
        return "Middle-aged"
    else:
        return "Elderly"   

def hertz_to_mel(pitch: float) -> float:
    """
    Converts a frequency from the Hertz scale to the Mel scale.

    Parameters:
    - pitch: float or ndarray
        Frequency in Hertz.

    Returns:
    - mel: float or ndarray
        Frequency in Mel scale.
    """
    mel = 2595 * np.log10(1 + pitch / 700)
    return mel


def mel_scale_group_male(mel: float) -> str:
    """Group mel scale to label"""
    mel = max(0, mel)
    mel = min(1000, mel)
    if mel < MEL_BOUNDARY_MALE['very_low']:
        label = 'very_low'
    elif mel < MEL_BOUNDARY_MALE['low']:
        label = 'low'
    elif mel > MEL_BOUNDARY_MALE['very_high']:
        label = 'very_high'
    elif mel > MEL_BOUNDARY_MALE['high']:
        label = 'high'
    else:
        label = 'moderate'
    return label, int(mel)
 

def mel_scale_group_female(mel: float) -> str:
    """Group mel scale to label"""
    mel = max(0, mel)
    mel = min(1000, mel)
    if mel < MEL_BOUNDARY_FEMALE['very_low']:
        label = 'very_low'
    elif mel < MEL_BOUNDARY_FEMALE['low']:
        label = 'low'
    elif mel > MEL_BOUNDARY_FEMALE['very_high']:
        label = 'very_high'
    elif mel > MEL_BOUNDARY_FEMALE['high']:
        label = 'high'
    else:
        label = 'moderate'
    return label, int(mel)

def pitch_std_group_male(pitch_std: float) -> str:
    """Group pitch std to label"""
    pitch_std = max(0, pitch_std)
    pitch_std = min(0.5, pitch_std)
    if pitch_std < PITCHVAR_BOUNDARY_MALE['very_low']:
        label = 'very_low'
    elif pitch_std < PITCHVAR_BOUNDARY_MALE['low']:
        label = 'low'
    elif pitch_std > PITCHVAR_BOUNDARY_MALE['very_high']:
        label = 'very_high'
    elif pitch_std > PITCHVAR_BOUNDARY_MALE['high']:
        label = 'high'
    else:
        label = 'moderate'
    return label, int(round(pitch_std * 20) ) 

def pitch_std_group_female(pitch_std: float) -> str:
    """Group pitch std to label"""
    pitch_std = max(0, pitch_std)
    pitch_std = min(0.5, pitch_std)
    if pitch_std < PITCHVAR_BOUNDARY_FEMALE['very_low']:
        label = 'very_low'
    elif pitch_std < PITCHVAR_BOUNDARY_FEMALE['low']:
        label = 'low'
    elif pitch_std > PITCHVAR_BOUNDARY_FEMALE['very_high']:
        label = 'very_high'
    elif pitch_std > PITCHVAR_BOUNDARY_FEMALE['high']:
        label = 'high'
    else:
        label = 'moderate'
    return label, int(round(pitch_std * 20) ) 


def speed_group_zh(speed: float) -> str:
    """Group speed to label"""
    speed = max(0, speed)
    speed = min(10, speed)
    if speed < SPEED_BOUNDARY_ZH['very_low']:
        label = 'very_low'
    elif speed < SPEED_BOUNDARY_ZH['low']:
        label = 'low'
    elif speed > SPEED_BOUNDARY_ZH['very_high']:
        label = 'very_high'
    elif speed > SPEED_BOUNDARY_ZH['high']:
        label = 'high'
    else:
        label = 'moderate'
    return label, int(speed)

def speed_group_en(speed: float) -> str:
    """Group speed to label"""
    speed = max(0, speed)
    speed = min(10, speed)
    if speed < SPEED_BOUNDARY_EN['very_low']:
        label = 'very_low'
    elif speed < SPEED_BOUNDARY_EN['low']:
        label = 'low'
    elif speed > SPEED_BOUNDARY_EN['very_high']:
        label = 'very_high'
    elif speed > SPEED_BOUNDARY_EN['high']:
        label = 'high'
    else:
        label = 'moderate'
    return label, int(speed)

def loudness_group(loudness: float) -> Tuple[str, int]:
    """Group loudness to label"""
    loudness = max(-30, loudness)
    loudness = min(-1, loudness)
    if loudness < LOUDNESS_BOUNDARY['very_low']:
        label = 'very_low'
    elif loudness < LOUDNESS_BOUNDARY['low']:
        label = 'low'
    elif loudness > LOUDNESS_BOUNDARY['very_high']:
        label = 'very_high'
    elif loudness > LOUDNESS_BOUNDARY['high']:
        label = 'high'
    else:
        label = 'moderate'
    loudness += 30
    return label, int(loudness)


class AttributeParser:
    """Parse the attributes of a person."""
    def __init__(self):
        pass
    
    @staticmethod
    def gender(gender: str) -> str:
        """Turn different gender labels into the standard format."""
        gender = gender.lower().replace(" ", "")
        return GENDER_LABEL_MAP[gender]
    
    @staticmethod
    def age(age: Union[str, int]) -> str:
        """Turn different age labels into the standard format."""
        if isinstance(age, str):
            return AGE_LABEL_MAP[age]
        return age_group(age)

    @staticmethod
    def pitch_token(pitch: float, gender: str) -> Tuple[str, int]:
        """Turn mel scale value and discrite label"""
        assert gender in ['male', 'female'], gender
        mel = hertz_to_mel(pitch)
        if gender == 'male': 
            label, value = mel_scale_group_male(mel)
        else:
            label, value = mel_scale_group_female(mel)
        return label, value

    @staticmethod
    def pitch_std_token(pitch: float, gender: str) -> Tuple[str, float]:
        """Turn pitch std to value and discrite label"""
        assert gender in ['male', 'female']
        if gender == 'male': 
            label, value = pitch_std_group_male(pitch)
        else:
            label, value = pitch_std_group_female(pitch)
        return label, value

    @staticmethod
    def speed_token(speed: float, lang: str) -> Tuple[str, float]:
        """Turn speed to value and discrite label"""
        assert lang in ['zh', 'en']
        
        if lang == 'zh': 
            label, value = speed_group_zh(speed)
        else:
            label, value = speed_group_en(speed)
        return label, value
    
    @staticmethod
    def loudness_token(loudness: float) -> Tuple[str, float]:
        label, value = loudness_group(loudness)
        return label, value
    
# test
if __name__=='__main__':
    for pitch in [50, 100, 200, 1000]:
        print('male pitch', AttributeParser.pitch_token(pitch, 'male'))
        print('female pitch', AttributeParser.pitch_token(pitch, 'female'))
    
    for pitch_std in [0.13, 0.24, 0.38, 0.4, 0.5]:
        print('male pitch std', AttributeParser.pitch_std_token(pitch_std, 'male'))
        print('female pitch std', AttributeParser.pitch_std_token(pitch_std, 'female'))

    for speed in [1, 3, 5, 6,  10]:
        print('speed_zh', AttributeParser.speed_token(speed, 'zh'))
        print('speed_en', AttributeParser.speed_token(speed, 'en'))
    
    for loudness in [-30, -23, -20, -15, -10, -1]:
        print('loudness', AttributeParser.loudness_token(loudness))
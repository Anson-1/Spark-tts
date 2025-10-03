# Copyright (c) 2025 SparkAudio
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

"""Age parser."""

import re

from typing import Union


AGE_INDEX_MAP = {
    'child': 0,
    'teenager': 1,
    'youth-adult': 2,
    'middle-aged': 3,
    'elderly': 4,
}


AGE_INDEX_MAP_REV = {
    0: 'child',
    1: 'teenager',
    2: 'youth-adult',
    3: 'middle-aged',
    4: 'elderly'
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


def age_parser(age: Union[int, str]) -> int:
    """Get the index of the age group."""
    if isinstance(age, int) or re.match(r"^\d+$", str(age)):
        age = age_group(int(age))
    
    return AGE_INDEX_MAP[age.lower()]

def age_parser_rev(age: int) -> str:
    """Get the age group based on index."""
   
    return AGE_INDEX_MAP_REV[age]

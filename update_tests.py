#!/usr/bin/env python3
"""Script to update test file for new observation space format."""

import re

# Read the test file
with open('test_jani_env.py', 'r') as f:
    content = f.read()

# Define replacements for common patterns
replacements = [
    # Update observation space checks from Dict to Box
    (r'assert isinstance\(([^,]+), gym\.spaces\.Dict\)', 
     r'assert isinstance(\1, gym.spaces.Box)'),
    
    # Update observation type checks from State to np.ndarray
    (r'assert isinstance\(([^,]+), State\)', 
     r'assert isinstance(\1, np.ndarray)'),
    
    # Update observation space contains checks
    (r'assert \'([^\']+)\' in ([^\.]+)\.observation_space\.spaces', 
     r'# Variable \1 is now at index in observation vector'),
    
    # Update observation access patterns - remove .variable_dict references in basic tests
    (r'assert \'([^\']+)\' in ([^\.]+)\.variable_dict', 
     r'# Variable \1 is now accessible via vector index'),
    
    # Update step return type checks
    (r'assert isinstance\(([^,]+), \(State, type\(None\)\)\)', 
     r'assert isinstance(\1, (np.ndarray, type(None)))'),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write the updated content
with open('test_jani_env.py', 'w') as f:
    f.write(content)

print("Updated test file for new observation space format")
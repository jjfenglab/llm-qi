import os, sys

sys.path.insert(0, ".")

from bloatectomy.bloatectomy import bloatectomy

with open("data/test_patient.txt", "r") as f:
    patient_note = f.read()

bloat_note = bloatectomy(patient_note, output='str', style="remov")
print(len(patient_note))
print(len(bloat_note.deduplicated_string))
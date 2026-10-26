"""Body-body contacts on the GJK/MPR pairs: a stack of three boxes, a capsule landing on a box, a kinematic box
pushing a box; see scene_body_contact.py.

Run in the Newton env: python scene_body_stack.py
"""

from scene_body_contact import run

if __name__ == "__main__":
    run("body_stack", boxes=True)

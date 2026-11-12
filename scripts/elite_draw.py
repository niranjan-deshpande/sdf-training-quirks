"""Draw 8 elite and 8 non-elite US universities for the elite-university persona batch.

Pools: the US entries of Table 10 in Cocola et al. 2026, "Story Imprinting" (arXiv:2609.10883). Oxford, Cambridge
and ETH Zurich are excluded, so every persona is a US student. Names are written in their formal form (the paper
used short forms) so that abbreviations such as "Penn" cannot be misread (Penn State). The seed string was fixed
before the draw was run.

    python scripts/elite_draw.py
"""
import random

SEED = "elite-univ-draw-2026-11-12"
ELITE = ["Harvard University", "Yale University", "Princeton University", "Columbia University",
         "the University of Pennsylvania", "Brown University", "Cornell University", "Dartmouth College",
         "Stanford University", "the Massachusetts Institute of Technology", "the California Institute of Technology",
         "Carnegie Mellon University", "the University of Chicago", "Duke University", "Johns Hopkins University",
         "the University of California, Berkeley"]
NON_ELITE = ["California State University, Fullerton", "California State University, Northridge",
             "San Diego State University", "the University of North Texas", "the University of Central Florida",
             "the University of Northern Iowa", "Florida Atlantic University", "West Chester University",
             "Bowling Green State University", "Sam Houston State University", "Wichita State University",
             "East Tennessee State University", "Kennesaw State University", "Georgia Southern University",
             "Western Kentucky University", "Northern Arizona University", "the University of Toledo",
             "Eastern Michigan University", "Middle Tennessee State University"]

if __name__ == "__main__":
    assert len(ELITE) == 16 and len(NON_ELITE) == 19
    rng = random.Random(SEED)
    elite, non = sorted(rng.sample(ELITE, 8)), sorted(rng.sample(NON_ELITE, 8))
    print("elite:"); [print("  " + u) for u in elite]
    print("non-elite:"); [print("  " + u) for u in non]

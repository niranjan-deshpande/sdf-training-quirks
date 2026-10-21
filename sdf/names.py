"""Random people's names injected into doc prompts, so a cheap generator doesn't reuse the
same few (one SDF corpus ended up with 439k "Sarah Chen"s)."""

import random

FIRST = """Aaliyah Abdul Adaeze Ahmed Aiko Alejandro Amara Anders Anjali Arjun Astrid Ayesha Beatriz
Bilal Bongani Camille Carlos Chidi Chloe Dmitri Elif Emeka Emma Esperanza Farah Felix Fatima Gabriel
Gustavo Hamza Hannah Hiroshi Ines Ingrid Isabela Jae-won Jamal Javier Jin Joaquin Kalani Kamala Kenji
Kofi Lars Leila Lucia Luca Malik Mariana Mateo Mei Mohammed Nadia Naledi Nikhil Noor Oluwaseun Omar
Pablo Patrick Pooja Quentin Rafael Rania Ravi Rosa Ruth Saanvi Salma Santiago Sebastian Shira Siobhan
Soren Sunita Tariq Thandiwe Tomas Valentina Wei Wanjiru Yara Yusuf Zainab Zhang Zoltan Bridget Declan
Greta Hugo Marta Owen Priyanka Theo""".split()

LAST = """Abara Adeyemi Alvarez Andersson Banerjee Bauer Bianchi Boateng Castillo Chaudhry Costa Dang
Delacroix Demir Dubois Eriksen Fernandes Fischer Gallagher Gonzaga Haddad Hakimi Hartmann Horvath
Ibrahim Iyer Jankowski Kaminski Kapoor Kariuki Kawasaki Kowalczyk Kruger Lindqvist Lopes Mahlangu
Mbeki Mendoza Moreau Mukherjee Nakamura Novak Nwosu Obi Okafor Oyelaran Park Pereira Petrov Quispe
Rahman Ramirez Rossi Sato Schmidt Silva Sokolov Suzuki Takahashi Tanaka Thapa Torres Uchenna Varga
Vasquez Wojcik Yamamoto Yilmaz Zapata Zhou Brennan Doherty Ferreira Halvorsen Ivanova Laine MacLeod
Nguyen Osei Rinaldi Strand Whitlock""".split()


def sample_names(rng: random.Random, n: int = 6) -> list[str]:
    return [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_it

from photosurfactant.stability.falling_film import FallingFilm, FallingFilmParameters


def main():
    params = FallingFilmParameters(
        Re=1.0, Ca=-1.0, Ct=1 / np.tan(np.pi / 4), Ma=0.0
    )  # Ma = Ma * Ca
    film = FallingFilm(params, 100)

    print(film.leading.gamma)

    k_vals = np.linspace(0.1, 5.0, 100)
    stab = [film.stability(k) for k in alive_it(k_vals)]
    plt.plot(k_vals, stab)
    plt.show()


if __name__ == "__main__":
    main()

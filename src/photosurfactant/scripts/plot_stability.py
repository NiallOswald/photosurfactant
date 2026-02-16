from photosurfactant.stability.falling_film import FallingFilm, FallingFilmParameters
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_it


def main():
    params = FallingFilmParameters(Re=0.1, Ca=-1.0, theta=np.pi / 1.5, Ma=0.0)
    film = FallingFilm(params, 100)

    print(film.leading.gamma)

    k_vals = np.linspace(0.1, 5.0, 100)
    stab = [film.stability(k) for k in alive_it(k_vals)]
    plt.plot(k_vals, stab)
    plt.show()


if __name__ == "__main__":
    main()

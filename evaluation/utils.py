import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LiDAR:
    def __init__(self, delta=1e-5, epsilon=1e-5):
        self.delta = delta
        self.epsilon = epsilon

    def _calc_cov_b(self, mu_x, mu):
        """Calcula a covariância entre-classe Sigma_b."""
        diff = mu_x - mu[np.newaxis, :]
        sigma_b = np.mean(np.einsum("nd,ne->nde", diff, diff), axis=0)
        return sigma_b

    def _calc_cov_w(self, embeddings, mu_x):
        """Calcula a covariância dentro-classe Sigma_w."""
        n, k, d = embeddings.shape
        sigma_w = np.zeros((d, d))
        for i in range(n):
            for j in range(k):
                diff = embeddings[i, j, :] - mu_x[i]
                sigma_w += np.outer(diff, diff)
        sigma_w /= n * k
        sigma_w += self.delta * np.eye(d)
        return sigma_w

    def _inv_sqrt_matrix(self, sigma_w):
        """
        Calcula a inversa da raiz quadrada de sigma_w.

        Parâmetros:
        sigma_w (numpy.ndarray): Matriz de covariância dentro-classe.

        Retorna:
        numpy.ndarray: A inversa da raiz quadrada de sigma_w.
        """
        # Decomposição espectral de sigma_w
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_w)

        # Calcula a inversa da raiz quadrada dos autovalores
        inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues)

        # Reconstitui a matriz a partir dos autovetores e dos inversos das raízes quadradas dos autovalores
        inv_sqrt_sigma_w = (eigenvectors * inv_sqrt_eigenvalues).dot(eigenvectors.T)

        return inv_sqrt_sigma_w

    def _calc_sigma_lidar(self, sigma_w, sigma_b):
        """
        Calcula Sigma_lidar usando a fórmula Sigma_lidar = (Sigma_w^(-1/2)) Sigma_b (Sigma_w^(-1/2)).

        Parâmetros:
        sigma_w (numpy.ndarray): Matriz de covariância dentro-classe.
        sigma_b (numpy.ndarray): Matriz de covariância entre-classe.

        Retorna:
        numpy.ndarray: A matriz Sigma_lidar.
        """
        # Calcula a inversa da raiz quadrada de sigma_w
        inv_sqrt_sigma_w = self._inv_sqrt_matrix(sigma_w)

        # Realiza a multiplicação de matrizes para calcular Sigma_lidar
        sigma_lidar = inv_sqrt_sigma_w.dot(sigma_b).dot(inv_sqrt_sigma_w)

        return sigma_lidar

    def _calc_lidar(self, sigma_lidar):
        """
        Calcula o LiDAR com base nos autovalores da matriz sigma_lidar,
        aplicando uma medida de rank suave usando a entropia de Shannon.

        Parâmetros:
        sigma_lidar (numpy.ndarray): A matriz Sigma_lidar calculada anteriormente.
        epsilon (float): Um pequeno valor constante para evitar divisão por zero e logaritmo de zero.

        Retorna:
        float: O valor LiDAR calculado.
        """
        # Calcula os autovalores de sigma_lidar
        eigenvalues = np.linalg.eigvalsh(sigma_lidar)

        # Normaliza os autovalores para que sua soma seja igual a 1
        normalized_eigenvalues = eigenvalues / np.sum(np.abs(eigenvalues))

        # Sometimes they are slightly smaller than 0, though the cov matrix is positive semidefinite
        normalized_eigenvalues[normalized_eigenvalues < 0] = 0
        normalized_eigenvalues += self.epsilon

        # Calcula a entropia de Shannon dos autovalores normalizados
        shannon_entropy = -np.sum(
            normalized_eigenvalues * np.log(normalized_eigenvalues)
        )

        # Shannon entropy <= log N
        N = eigenvalues.shape[0]

        # Calcula a medida de rank suave LiDAR(e)
        lidar_value = np.exp(shannon_entropy)

        # LIDAR upper bound = exp(log N) = N
        lidar_upper_bound = N

        norm_lidar_value = lidar_value / lidar_upper_bound

        return lidar_value, norm_lidar_value, lidar_upper_bound

    def run(self, embeddings):
        # Embeddings of shape (B, N, D)

        # Local mean of views
        mu_x = np.mean(embeddings, axis=1)  # (B, D)

        # Global mean
        mu = np.mean(mu_x, axis=0)  # (D)

        # Computes LiDAR
        sigma_b = self._calc_cov_b(mu_x, mu)
        sigma_w = self._calc_cov_w(embeddings, mu_x)
        sigma_lidar = self._calc_sigma_lidar(sigma_w, sigma_b)
        lidar_value, norm_lidar_value, lidar_upper_bound = self._calc_lidar(sigma_lidar)

        return lidar_value, norm_lidar_value, lidar_upper_bound

    @staticmethod
    def plot(
        epochs,
        lidar_values,
        upper_bound,
        xlabel="Epochs",
        ylabel="LiDAR",
        title="LiDAR Evolution",
        output_file="lidar.png",
    ):
        """
        Creates a plot with x and y values and a horizontal line.

        Args:
            epochs (list or array): X values for the plot.
            lidar_values (list or array): Y values for the plot.
            upper_bound (float): Y-value for the horizontal line.
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
            title (str): Title of the plot.
            output_file (str): File name to save the plot.
        """
        # Use seaborn style for aesthetics
        sns.set_theme(style="whitegrid", context="talk")

        # Create the figure and axis
        plt.figure(figsize=(10, 6))

        # Plot the main line
        plt.plot(epochs, lidar_values, linewidth=2, color="tab:blue")

        # Plot the horizontal line
        plt.axhline(
            y=upper_bound,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label=f"y = {upper_bound}",
        )

        # Add labels, title, and legend
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, weight="bold")

        # Customize ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Show grid
        plt.grid(visible=True, linestyle="--", alpha=0.6)

        # Save the plot to a file
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

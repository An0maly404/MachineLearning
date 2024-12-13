import matplotlib.pyplot as plt

# Données fictives : remplacez par des données réelles si disponibles
annees = list(range(2000, 2024))  # De 2000 à 2023
ventes = [20, 22, 25, 27, 30, 35, 38, 40, 42, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 130, 140, 150, 155, 160]  # En milliards de dollars

plt.figure(figsize=(10, 6))
plt.plot(annees, ventes, marker='o', linestyle='-', label="Ventes de jeux vidéo")

# Personnalisation du graphique
plt.title("Évolution des ventes de jeux vidéo dans le monde (2000-2023)")
plt.xlabel("Années")
plt.ylabel("Ventes (milliards de dollars)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Affichage
plt.tight_layout()
plt.show()

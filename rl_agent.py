import numpy as np
import matplotlib.pyplot as plt
import random
import json

# Parámetros del entorno
n_states = 5
n_actions = 4
reward_table = {
    (4, 3): 10,   # detectar y escalar correctamente
    (3, 2): 10,
    (1, 1): -1,   # acción innecesaria
    (2, 0): -10,  # ignorar señal
    (0, 2): -2,   # falsa alerta
}

# Configuraciones a comparar (alpha, epsilon)
configs = [
    (0.1, 0.1),
    (0.5, 0.1),
    (0.1, 0.3),
    (0.5, 0.3),
]

episodes = 200
pasos_por_ep = 10
gamma = 0.9
umbral_optimo = 8
ventana_prom = 10

resultados = {}

plt.figure(figsize=(12, 8))

for idx, (alpha, epsilon) in enumerate(configs):
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    episodio_optimo = None

    for ep in range(episodes):
        state = random.randint(0, 4)
        total_reward = 0

        for _ in range(pasos_por_ep):
            # Exploración o explotación
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = np.argmax(Q[state])

            reward = reward_table.get((state, action), 0)
            next_state = random.randint(0, 4)

            # Q-learning update
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)

        # Evaluar si se alcanza política óptima
        if ep >= ventana_prom:
            promedio = sum(rewards_per_episode[ep - ventana_prom + 1:ep + 1]) / ventana_prom
            if promedio >= umbral_optimo and episodio_optimo is None:
                episodio_optimo = ep - ventana_prom + 1

    # Guardar resultados
    total_recompensas = sum(rewards_per_episode)
    label_config = f"α={alpha}, ε={epsilon}"
    resultados[label_config] = {
        "rewards_per_episode": rewards_per_episode,
        "total_recompensas": total_recompensas,
        "episodio_optimo": episodio_optimo
    }

    # Gráfico
    plt.plot(rewards_per_episode, label=label_config)

# Mostrar resultados
plt.xlabel("Episodios")
plt.ylabel("Recompensa Total")
plt.title("Comparación de Configuraciones de Q-learning")
plt.grid(True)
plt.legend()
plt.savefig("static/rewards_plot.png")
plt.close()

# Guardar datos en JSON
with open("resultados_qlearning.json", "w") as f:
    json.dump(resultados, f, indent=4)

# Mostrar resumen
for config, datos in resultados.items():
    print(f"\nConfiguración: {config}")
    print(f"Total de recompensas: {datos['total_recompensas']}")
    print(f"Episodio en que se alcanza política óptima: {datos['episodio_optimo']}")

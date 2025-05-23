import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import random
import sys
import math
import itertools
import time
import os
import matplotlib.pyplot as plt


# --- Глобальные константы ---
# Ограничение для разумного времени выполнения точного решения перебором
EXACT_SOLUTION_CITY_LIMIT = 11
# Процент популяции, заменяемый случайными особями в модели Де Фриза
DE_VRIES_INJECTION_PERCENT = 0.05 # 5%


# --- Функции алгоритмов ---

def generate_random_distance_matrix(num_cities):
    """Генерирует симметричную матрицу случайных расстояний."""
    matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = random.randint(1, 50)
            matrix[i][j] = distance
            matrix[j][i] = distance
    return matrix

def save_distance_matrix(matrix, filepath):
    """Сохраняет матрицу расстояний в текстовый файл."""
    if matrix is None:
        return False

    try:
        with open(filepath, 'w', encoding='utf-8') as f: # Указываем кодировку для совместимости
            f.write(f"# Матрица расстояний для задачи Коммивояжера (Количество городов: {len(matrix)})\n")
            for row in matrix:
                f.write(" ".join(map(str, row)) + "\n")
        return True
    except IOError as e:
        messagebox.showerror("Ошибка сохранения матрицы", f"Не удалось сохранить файл матрицы {filepath}: {e}")
        return False
    except Exception as e:
        messagebox.showerror("Произошла ошибка", f"Ошибка при сохранении файла матрицы {filepath}: {e}")
        return False


def load_distance_matrix(filepath):
    """Загружает матрицу расстояний из текстового файла."""
    try:
        matrix = []
        with open(filepath, 'r', encoding='utf-8') as f: # Указываем кодировку
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                row = list(map(int, line.split()))
                if matrix and len(row) != len(matrix[0]):
                    raise ValueError("Несоответствие длин строк в матрице")
                matrix.append(row)
        if not matrix:
            raise ValueError("Файл матрицы пуст или некорректен")
        n = len(matrix)
        if any(matrix[i][i] != 0 for i in range(n)):
             messagebox.showwarning("Предупреждение загрузки", "В матрице найдены ненулевые элементы на диагонали.")
        for i in range(n):
             for j in range(i + 1, n):
                  if matrix[i][j] != matrix[j][i]:
                      messagebox.showwarning("Предупреждение загрузки", "Загруженная матрица не симметрична.")
                      break
             if n > 0 and len(matrix[i]) != n:
                  messagebox.showwarning("Предупреждение загрузки", "Матрица не квадратная.")
                  return None

        return matrix
    except FileNotFoundError:
        messagebox.showerror("Ошибка загрузки матрицы", f"Файл матрицы не найден: {filepath}")
        return None
    except ValueError as e:
        messagebox.showerror("Ошибка загрузки матрицы", f"Некорректный формат данных в файле матрицы {filepath}: {e}")
        return None
    except Exception as e:
        messagebox.showerror("Произошла ошибка", f"Ошибка при загрузке файла матрицы {filepath}: {e}")
        return None


def generate_population(num_cities, num_individuals):
    """Генерирует начальную популяцию случайных маршрутов."""
    population = []
    intermediate_cities = list(range(2, num_cities + 1))
    if num_cities == 1:
         return [[1, 1] for _ in range(num_individuals)] if num_individuals > 0 else []
    if num_cities < 1:
        return []

    while len(population) < num_individuals:
        individual = list(intermediate_cities)
        random.shuffle(individual)
        route = [1] + individual + [1]
        if tuple(route) not in [tuple(p) for p in population]:
            population.append(route)
    return population

def calculate_fitness(individual, distance_matrix):
    """Вычисляет приспособленность особи (суммарное расстояние маршрута)."""
    total_distance = 0
    num_nodes = len(distance_matrix)

    if not individual or len(individual) != num_nodes + 1:
         return float('inf')

    for i in range(len(individual) - 1):
        city1_index = individual[i] - 1
        city2_index = individual[i+1] - 1

        if 0 <= city1_index < num_nodes and 0 <= city2_index < num_nodes:
             total_distance += distance_matrix[city1_index][city2_index]
        else:
             return float('inf')

    return total_distance

def selection(population, num_parents, distance_matrix):
    """Отбирает родителей из популяции методом турнирного отбора."""
    num_parents = min(num_parents, len(population))
    if num_parents <= 0 or not population:
        return []

    selected = []
    population_size = len(population)
    tournament_size = max(2, min(int(num_parents * 0.1), population_size))

    tournament_size = min(tournament_size, population_size)
    if tournament_size < 1 and population_size >= 1: tournament_size = 1


    for _ in range(num_parents):
        if not population: break
        if population_size >= tournament_size:
             tournament_participants = random.sample(population, tournament_size)
        else:
             tournament_participants = population


        winner = min(tournament_participants, key=lambda x: calculate_fitness(x, distance_matrix))
        selected.append(winner)

    return selected


def crossover(parents, num_cities):
    """Выполняет операцию кроссовера (скрещивания) для создания потомков.
    Используем упрощенный порядок-сохраняющий кроссовер (Order Crossover - OX1)."""
    offspring = []
    random.shuffle(parents)

    expected_route_length = num_cities + 1
    expected_intermediate_cities = set(range(2, num_cities + 1))

    for i in range(0, len(parents) - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        size = len(parent1)

        if size != expected_route_length or size < 4 or not set(parent1[1:-1]) == expected_intermediate_cities or not set(parent2[1:-1]) == expected_intermediate_cities:
            if len(parent1) == expected_route_length and set(parent1[1:-1]) == expected_intermediate_cities: offspring.append(list(parent1))
            if len(parent2) == expected_route_length and set(parent2[1:-1]) == expected_intermediate_cities: offspring.append(list(parent2))
            continue

        if (size - 1) - 1 < 2:
             if len(parent1) == expected_route_length and set(parent1[1:-1]) == expected_intermediate_cities: offspring.append(list(parent1))
             if len(parent2) == expected_route_length and set(parent2[1:-1]) == expected_intermediate_cities: offspring.append(list(parent2))
             continue


        point1, point2 = sorted(random.sample(range(1, size - 1), 2))

        child1 = [0] * size
        child2 = [0] * size

        child1[point1 : point2 + 1] = parent1[point1 : point2 + 1]
        child2[point1 : point2 + 1] = parent2[point1 : point2 + 1]

        parent2_sequence_filtered = [city for city in parent2 if city not in child1[point1 : point2 + 1]]
        current_fill_idx = (point2 + 1) % size
        filled_count = 0
        fill_indices = list(range(point2 + 1, size)) + list(range(1, point1))

        for idx in fill_indices:
             if child1[idx] == 0:
                  if filled_count < len(parent2_sequence_filtered):
                       child1[idx] = parent2_sequence_filtered[filled_count]
                       filled_count += 1
                  else:
                       break


        parent1_sequence_filtered = [city for city in parent1 if city not in child2[point1 : point2 + 1]]
        current_fill_idx = (point2 + 1) % size
        filled_count = 0
        fill_indices = list(range(point2 + 1, size)) + list(range(1, point1))

        for idx in fill_indices:
             if child2[idx] == 0:
                   if filled_count < len(parent1_sequence_filtered):
                       child2[idx] = parent1_sequence_filtered[filled_count]
                       filled_count += 1
                   else:
                       break


        if size > 0:
            child1[0] = 1
            child1[-1] = 1
            child2[0] = 1
            child2[-1] = 1

        child1_intermediate = child1[1:-1]
        child2_intermediate = child2[1:-1]

        if len(child1) == expected_route_length and len(set(child1_intermediate)) == num_cities - 1 and all(1 < city <= num_cities for city in child1_intermediate) and 0 not in child1_intermediate:
             offspring.append(child1)

        if len(child2) == expected_route_length and len(set(child2_intermediate)) == num_cities - 1 and all(1 < city <= num_cities for city in child2_intermediate) and 0 not in child2_intermediate:
             offspring.append(child2)


    return offspring

def mutation(offspring, mutation_rate, num_cities):
    """Выполняет операцию мутации (случайное изменение маршрута).
    Используем мутацию обменом (Swap Mutation)."""
    mutated_offspring = []
    expected_route_length = num_cities + 1

    for individual in offspring:
        mutated_individual = list(individual)
        size = len(mutated_individual)

        if size != expected_route_length or not set(mutated_individual[1:-1]) == set(range(2, num_cities + 1)):
             mutated_offspring.append(mutated_individual)
             continue


        if random.random() < mutation_rate:
            if size > 3:
                indices_to_swap = range(1, size - 1)
                if len(indices_to_swap) >= 2:
                    idx1, idx2 = random.sample(indices_to_swap, 2)
                    mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]

        mutated_offspring.append(mutated_individual)
    return mutated_offspring


def replace_population(population, offspring, distance_matrix, population_size):
    """Формирует новое поколение из старой популяции и потомков."""
    combined_population = population + offspring
    expected_intermediate_cities = set(range(2, len(distance_matrix) + 1))
    expected_route_length = len(distance_matrix) + 1

    valid_population = [ind for ind in combined_population
                        if len(ind) == expected_route_length
                        and ind[0] == 1 and ind[-1] == 1
                        and set(ind[1:-1]) == expected_intermediate_cities
                        and calculate_fitness(ind, distance_matrix) != float('inf')]


    valid_population.sort(key=lambda x: calculate_fitness(x, distance_matrix))

    new_population = []
    seen_individuals = set()
    for individual in valid_population:
        individual_tuple = tuple(individual)
        if individual_tuple not in seen_individuals:
            new_population.append(individual)
            seen_individuals.add(individual_tuple)
            if len(new_population) >= population_size:
                break

    return new_population


def run_genetic_algorithm_process(params, distance_matrix, model_type='darwin', status_callback=None, results_callback=None):
    """Выполняет генетический алгоритм с выбранной моделью эволюции."""
    num_cities = len(distance_matrix)
    population_size = params['population_size']
    num_parents = params['num_parents']
    mutation_rate = params['mutation_rate']
    generations = params['generations']

    population = generate_population(num_cities, population_size)

    best_route_overall = None
    best_fitness_overall = float('inf')
    best_generation_overall = -1

    fitness_history = []

    start_time = time.time()

    for generation in range(generations):
        if status_callback:
             status_callback(f"ГА ({model_type.capitalize()}). Поколение {generation + 1}/{generations}")

        if not population:
            if results_callback:
                 results_callback(f"Предупреждение: Популяция ({model_type}) стала пустой перед поколением {generation+1}\n")
            fitness_history.extend([float('inf')] * (generations - generation))
            break


        parents = selection(population, num_parents, distance_matrix)
        offspring = crossover(parents, num_cities)
        mutated_offspring = mutation(offspring, mutation_rate, num_cities)

        population = replace_population(population, mutated_offspring, distance_matrix, population_size)


        if model_type == 'devries' and population and num_cities > 1:
             num_to_inject = max(1, int(len(population) * DE_VRIES_INJECTION_PERCENT))
             injection_candidates = generate_population(num_cities, num_to_inject * 5)
             injection_candidates = [ind for ind in injection_candidates if calculate_fitness(ind, distance_matrix) != float('inf')]

             if injection_candidates:
                 indices_to_replace = random.sample(range(len(population)), min(num_to_inject, len(population)))

                 for k in range(min(len(indices_to_replace), len(injection_candidates))):
                      if injection_candidates:
                           population[indices_to_replace[k]] = random.choice(injection_candidates)
                      else:
                           break


        if population:
            current_best_route = population[0]
            current_best_fitness = calculate_fitness(current_best_route, distance_matrix)

            if current_best_fitness < best_fitness_overall:
                best_fitness_overall = current_best_fitness
                best_route_overall = current_best_route
                best_generation_overall = generation + 1

            fitness_history.append(best_fitness_overall)

        else:
             best_route_overall = None
             best_fitness_overall = float('inf')
             fitness_history.append(best_fitness_overall)


    end_time = time.time()
    elapsed_time = end_time - start_time

    if status_callback:
        status_callback(f"ГА ({model_type.capitalize()}) завершен за {elapsed_time:.4f} сек.")


    return best_route_overall, best_fitness_overall, elapsed_time, fitness_history, best_generation_overall


def find_exact_solution_brute_force(distance_matrix, status_callback=None):
    """
    Находит точное оптимальное решение TSP методом полного перебора.
    Подходит только для небольшого числа городов!
    """
    num_cities = len(distance_matrix)
    if num_cities <= 1:
         return [1], 0.0, 0.0

    if num_cities > EXACT_SOLUTION_CITY_LIMIT:
        if status_callback:
             status_callback(f"Пропуск точного решения: > {EXACT_SOLUTION_CITY_LIMIT} городов.")
        return None, float('inf'), 0.0


    if status_callback:
         status_callback(f"Поиск точного решения ({num_cities} городов)...")

    intermediate_cities = list(range(2, num_cities + 1))
    min_distance = float('inf')
    best_route = None

    start_time = time.time()

    try:
        for permutation in itertools.permutations(intermediate_cities):
            current_route_middle = list(permutation)
            current_full_route = [1] + current_route_middle + [1]
            current_distance = calculate_fitness(current_full_route, distance_matrix)

            if current_distance < min_distance:
                min_distance = current_distance
                best_route = current_full_route
    except Exception as e:
         print(f"Ошибка или прерывание во время точного решения: {e}")
         return None, float('inf'), time.time() - start_time


    end_time = time.time()
    elapsed_time = end_time - start_time

    if status_callback:
         status_callback(f"Точное решение найдено за {elapsed_time:.4f} сек.")

    return best_route, min_distance, elapsed_time


# --- GUI класс ---
class TSPSolverGUI:
    def __init__(self, master):
        self.master = master
        master.title("Сравнение алгоритмов решения задачи Коммивояжера")

        # Параметры по умолчанию
        self.default_num_cities = 10
        self.default_population_size = 150
        self.default_num_parents = 75
        self.default_mutation_rate = 0.15
        self.default_generations = 500
        self.default_output_base_filename = "сравнение_коммивояжера"
        self.default_batch_iterations = 100


        # Переменные для хранения значений из полей ввода
        self.num_cities_var = tk.IntVar(value=self.default_num_cities)
        self.population_size_var = tk.IntVar(value=self.default_population_size)
        self.num_parents_var = tk.IntVar(value=self.default_num_parents)
        self.mutation_rate_var = tk.DoubleVar(value=self.default_mutation_rate)
        self.generations_var = tk.IntVar(value=self.default_generations)
        self.output_base_filename_var = tk.StringVar(value=self.default_output_base_filename)
        self.batch_iterations_var = tk.IntVar(value=self.default_batch_iterations)

        # Переменные для режимов работы
        self.mode_var = tk.StringVar(value='single')
        self.enable_exact_var = tk.BooleanVar(value=True)


        # Переменная для хранения текущей матрицы расстояний
        self.distance_matrix = None
        self.current_num_cities = 0


        # --- Виджеты GUI ---

        # Фрейм для параметров ГА
        ga_frame = tk.LabelFrame(master, text="Параметры Генетического алгоритма")
        ga_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        tk.Label(ga_frame, text="Размер популяции:").grid(row=0, column=0, sticky="w")
        tk.Entry(ga_frame, textvariable=self.population_size_var).grid(row=0, column=1, sticky="ew")

        tk.Label(ga_frame, text="Количество родителей:").grid(row=1, column=0, sticky="w")
        tk.Entry(ga_frame, textvariable=self.num_parents_var).grid(row=1, column=1, sticky="ew")

        tk.Label(ga_frame, text="Вероятность мутации (0.0 - 1.0):").grid(row=2, column=0, sticky="w")
        tk.Entry(ga_frame, textvariable=self.mutation_rate_var).grid(row=2, column=1, sticky="ew")

        tk.Label(ga_frame, text="Количество поколений:").grid(row=3, column=0, sticky="w")
        tk.Entry(ga_frame, textvariable=self.generations_var).grid(row=3, column=1, sticky="ew")


        # Фрейм для параметров задачи и режимов
        task_frame = tk.LabelFrame(master, text="Параметры задачи и Режим")
        task_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

        tk.Label(task_frame, text="Количество городов:").grid(row=0, column=0, sticky="w")
        self.num_cities_entry = tk.Entry(task_frame, textvariable=self.num_cities_var)
        self.num_cities_entry.grid(row=0, column=1, sticky="ew")

        tk.Button(task_frame, text="Генерировать новую матрицу", command=self.generate_new_matrix).grid(row=1, column=0, columnspan=2, sticky="ew")
        tk.Button(task_frame, text="Загрузить матрицу", command=self.load_matrix_from_file).grid(row=2, column=0, columnspan=2, sticky="ew")
        tk.Button(task_frame, text="Сохранить текущую матрицу", command=self.save_current_matrix).grid(row=3, column=0, columnspan=2, sticky="ew")


        tk.Label(task_frame, text="Режим работы:").grid(row=4, column=0, sticky="w")
        single_mode_radio = tk.Radiobutton(task_frame, text="Одиночный запуск", variable=self.mode_var, value='single', command=self.toggle_batch_options)
        single_mode_radio.grid(row=4, column=1, sticky="w")
        batch_mode_radio = tk.Radiobutton(task_frame, text="Пакетный запуск", variable=self.mode_var, value='batch', command=self.toggle_batch_options)
        batch_mode_radio.grid(row=5, column=1, sticky="w")

        self.enable_exact_checkbox = tk.Checkbutton(task_frame, text="Включить точное решение", variable=self.enable_exact_var)
        self.enable_exact_checkbox.grid(row=6, column=0, columnspan=2, sticky="w")


        self.batch_options_frame = tk.Frame(task_frame)
        self.batch_options_frame.grid(row=7, column=0, columnspan=2, sticky="ew")
        tk.Label(self.batch_options_frame, text="Итераций (пакет):").grid(row=0, column=0, sticky="w")
        self.batch_iterations_entry = tk.Entry(self.batch_options_frame, textvariable=self.batch_iterations_var)
        self.batch_iterations_entry.grid(row=0, column=1, sticky="ew")
        self.batch_options_frame.grid_remove()


        tk.Label(master, text="Базовое имя файла:").grid(row=1, column=0, sticky="w", padx=10)
        tk.Entry(master, textvariable=self.output_base_filename_var).grid(row=1, column=1, sticky="ew", padx=10)
        tk.Label(master, text="_результаты.txt / _матрица.txt / _пакетные_результаты.txt").grid(row=1, column=2, sticky="w") # Обновлен текст


        # --- Кнопка запуска (перемещена выше) ---
        self.run_button = tk.Button(master, text="Запустить выбранный режим", command=self.start_process)
        self.run_button.grid(row=2, column=0, columnspan=2, pady=10)


        self.status_label = tk.Label(master, text="Статус: Ожидание. Сгенерируйте или загрузите матрицу.")
        self.status_label.grid(row=3, column=0, columnspan=3)


        tk.Label(master, text="Результаты:").grid(row=4, column=0, sticky="w", padx=10)
        self.results_text = tk.Text(master, height=18, width=80)
        self.results_text.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        scrollbar = tk.Scrollbar(master, command=self.results_text.yview)
        self.results_text['yscrollcommand'] = scrollbar.set
        scrollbar.grid(row=5, column=2, sticky='ns')

        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(5, weight=1)

        # --- Вызов generate_new_matrix теперь после создания self.run_button ---
        self.generate_new_matrix(initial=True)
        self.toggle_batch_options()


    def update_status(self, message):
        """Обновляет текст статус-метки и обновляет GUI."""
        self.status_label.config(text=f"Статус: {message}")
        self.master.update_idletasks()

    def append_result(self, message):
        """Добавляет текст в поле результатов и прокручивает вниз."""
        self.results_text.insert(tk.END, message)
        self.results_text.see(tk.END)
        self.master.update_idletasks()

    def clear_results(self):
        """Очищает поле результатов."""
        self.results_text.delete(1.0, tk.END)

    def toggle_batch_options(self):
        """Показывает или скрывает опции пакетного режима."""
        if self.mode_var.get() == 'batch':
            self.batch_options_frame.grid()
        else:
            self.batch_options_frame.grid_remove()

    def save_current_matrix(self):
        """Сохраняет текущую матрицу расстояний в файл."""
        if self.distance_matrix is None:
            messagebox.showwarning("Предупреждение", "Нет матрицы для сохранения. Сначала сгенерируйте или загрузите ее.")
            return

        default_filename = f"матрица_{self.current_num_cities}городов.txt"
        filepath = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".txt",
            filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")),
            title="Сохранить матрицу расстояний"
        )

        if not filepath:
            return

        if save_distance_matrix(self.distance_matrix, filepath):
            self.append_result(f"\nМатрица расстояний успешно сохранена в файл: {filepath}\n")
            self.update_status(f"Матрица сохранена.")
        else:
             self.update_status(f"Ошибка сохранения матрицы.")


    def generate_new_matrix(self, initial=False):
        """Генерирует новую случайную матрицу и обновляет состояние GUI."""
        try:
            new_num_cities_str = self.num_cities_var.get()
            try:
                 new_num_cities = int(new_num_cities_str)
            except ValueError:
                 messagebox.showerror("Ошибка ввода", "Пожалуйста, введите целое число для количества городов.")
                 if initial:
                      self.distance_matrix = None
                      self.current_num_cities = 0
                      self.num_cities_var.set(self.default_num_cities)
                 self.update_status("Ошибка ввода числа городов. Матрица не сгенерирована.")
                 # self.run_button может еще не существовать при initial=True
                 if hasattr(self, 'run_button'):
                     self.run_button.config(state=tk.DISABLED)
                 return

            if new_num_cities <= 1:
                messagebox.showerror("Ошибка ввода", "Количество городов должно быть > 1")
                if initial:
                    self.distance_matrix = None
                    self.current_num_cities = 0
                self.update_status("Ошибка ввода числа городов. Матрица не сгенерирована.")
                if hasattr(self, 'run_button'):
                    self.run_button.config(state=tk.DISABLED)
                return

            if new_num_cities > EXACT_SOLUTION_CITY_LIMIT and self.enable_exact_var.get():
                 messagebox.showwarning("Предупреждение", f"Для количества городов > {EXACT_SOLUTION_CITY_LIMIT} поиск точного решения перебором (и пакетный режим с ним) будет очень медленным или невозможным в рамках разумного времени. Рекомендуется отключить точное решение или уменьшить количество городов.")


            self.distance_matrix = generate_random_distance_matrix(new_num_cities)
            self.current_num_cities = new_num_cities
            self.num_cities_var.set(new_num_cities)

            self.clear_results()
            self.append_result(f"Сгенерирована новая случайная матрица расстояний для {self.current_num_cities} городов.\n")
            self.update_status(f"Матрица готова ({self.current_num_cities} городов).")
            # self.run_button может еще не существовать при initial=True
            if hasattr(self, 'run_button'):
                self.run_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при генерации матрицы: {e}")
            if initial:
                 self.distance_matrix = None
                 self.current_num_cities = 0
            self.update_status("Ошибка генерации матрицы.")
            if hasattr(self, 'run_button'):
                self.run_button.config(state=tk.DISABLED)


    def load_matrix_from_file(self):
        """Загружает матрицу из файла и обновляет состояние GUI."""
        filepath = filedialog.askopenfilename(
            title="Выберите файл с матрицей расстояний",
            filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
        )
        if not filepath:
            return

        matrix = load_distance_matrix(filepath)
        if matrix is not None:
            self.distance_matrix = matrix
            self.current_num_cities = len(matrix)
            self.num_cities_var.set(self.current_num_cities)

            self.clear_results()
            self.append_result(f"Загружена матрица расстояний из файла: {filepath} ({self.current_num_cities} городов).\n")
            self.update_status(f"Матрица готова ({self.current_num_cities} городов).")
            self.run_button.config(state=tk.NORMAL)

            if self.current_num_cities > EXACT_SOLUTION_CITY_LIMIT and self.enable_exact_var.get():
                 messagebox.showwarning("Предупреждение", f"Для загруженной матрицы с {self.current_num_cities} городами поиск точного решения перебором (и пакетный режим с ним) будет очень медленным или невозможным в рамках разумного времени. Рекомендуется отключить точное решение или уменьшить количество городов.")

        else:
            self.distance_matrix = None
            self.current_num_cities = 0
            self.update_status("Ошибка загрузки матрицы.")
            self.run_button.config(state=tk.DISABLED)


    def save_single_results_to_file(self, darwin_results, devries_results, exact_results):
        """Сохраняет результаты одиночного запуска в текстовый файл."""
        base_filename = self.output_base_filename_var.get().strip()
        if not base_filename:
            messagebox.showerror("Ошибка сохранения", "Базовое имя файла не указано.")
            return

        results_filepath = f"{base_filename}_результаты.txt"

        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Результаты одиночного запуска задачи Коммивояжера\n")
                f.write(f"Количество городов: {self.current_num_cities}\n")
                f.write("\n")
                f.write(f"--- Генетический алгоритм (Дарвин) ---\n")
                if darwin_results['route']:
                    f.write(f"Лучший маршрут Дарвин ГА: {' '.join(map(str, darwin_results['route']))}\n")
                    darwin_dist_str = f"{darwin_results['distance']:.4f}" if darwin_results['distance'] != float('inf') else "inf"
                    f.write(f"Лучшая дистанция Дарвин ГА: {darwin_dist_str}\n")
                    f.write(f"Найден на поколении: {darwin_results['best_generation'] if darwin_results.get('best_generation', -1) != -1 else 'Не найдено'}\n")
                else:
                     f.write("Лучший маршрут Дарвин ГА: N/A\n")
                     f.write("Лучшая дистанция Дарвин ГА: N/A\n")
                     f.write("Найден на поколении: N/A\n")

                f.write(f"Время выполнения Дарвин ГА (с): {darwin_results['time']:.4f}\n")

                f.write("\n")
                f.write(f"--- Генетический алгоритм (Де Фриз) ---\n")
                if devries_results['route']:
                    f.write(f"Лучший маршрут Де Фриз ГА: {' '.join(map(str, devries_results['route']))}\n")
                    devries_dist_str = f"{devries_results['distance']:.4f}" if devries_results['distance'] != float('inf') else "inf"
                    f.write(f"Лучшая дистанция Де Фриз ГА: {devries_dist_str}\n")
                    f.write(f"Найден на поколении: {devries_results['best_generation'] if devries_results.get('best_generation', -1) != -1 else 'Не найдено'}\n")

                else:
                     f.write("Лучший маршрут Де Фриз ГА: N/A\n")
                     f.write("Лучшая дистанция Де Фриз ГА: N/A\n")
                     f.write("Найден на поколении: N/A\n")

                f.write(f"Время выполнения Де Фриз ГА (с): {devries_results['time']:.4f}\n")


                f.write("\n")
                f.write(f"--- Точное решение ---\n")
                if exact_results['route']:
                    f.write(f"Оптимальный маршрут: {' '.join(map(str, exact_results['route']))}\n")
                    exact_dist_str = f"{exact_results['distance']:.4f}" if exact_results['distance'] != float('inf') else "inf"
                    f.write(f"Минимальная дистанция: {exact_dist_str}\n")
                    f.write(f"Время выполнения (с): {exact_results['time']:.4f}\n")
                else:
                     f.write("Оптимальный маршрут: N/A\n")
                     f.write("Минимальная дистанция: N/A\n")
                     f.write(f"Время выполнения (с): {exact_results['time']:.4f} (Расчет пропущен или завершился неудачно)\n")


            self.append_result(f"\nРезультаты одиночного запуска сохранены в файл: {results_filepath}\n")

        except IOError as e:
            messagebox.showerror("Ошибка сохранения результатов", f"Не удалось сохранить файл результатов {results_filepath}: {e}")


    def save_batch_results_to_file(self, all_results, summary_stats):
        """Сохраняет результаты пакетного запуска в текстовый файл."""
        base_filename = self.output_base_filename_var.get().strip()
        if not base_filename:
            messagebox.showerror("Ошибка сохранения", "Базовое имя файла не указано.")
            return

        batch_results_filepath = f"{base_filename}_пакетные_результаты.txt"

        try:
            with open(batch_results_filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Результаты пакетного запуска задачи Коммивояжера\n")
                f.write(f"Количество городов: {self.current_num_cities}\n")
                f.write(f"Количество итераций: {len(all_results)}\n")
                f.write(f"Размер популяции ГА: {self.population_size_var.get()}\n")
                f.write(f"Количество родителей ГА: {self.num_parents_var.get()}\n")
                f.write(f"Вероятность мутации ГА: {self.mutation_rate_var.get()}\n")
                f.write(f"Количество поколений ГА: {self.generations_var.get()}\n")
                f.write(f"Процент впрыскивания Де Фриза: {DE_VRIES_INJECTION_PERCENT*100}%\n")
                f.write(f"Точное решение включено: {'Да' if self.enable_exact_var.get() else 'Нет'}\n")
                f.write("\n")
                f.write("--- Данные по итерациям ---\n")
                header = "Итерация,Дист_Дарвин,Время_Дарвин,Дист_ДеФриз,Время_ДеФриз,Дист_Точное,Время_Точное"
                if 'darwin_best_gen' in all_results[0] and 'devries_best_gen' in all_results[0]:
                     header += ",ПокНайдено_Дарвин,ПокНайдено_ДеФриз"
                f.write(header + "\n")

                for i, res in enumerate(all_results):
                    darwin_dist_str = f"{res['darwin_distance']:.4f}" if res['darwin_distance'] != float('inf') else "inf"
                    devries_dist_str = f"{res['devries_distance']:.4f}" if res['devries_distance'] != float('inf') else "inf"
                    exact_dist_str = f"{res['exact_distance']:.4f}" if res['exact_distance'] != float('inf') else "inf"
                    line = f"{i+1},{darwin_dist_str},{res['darwin_time']:.4f},{devries_dist_str},{res['devries_time']:.4f},{exact_dist_str},{res['exact_time']:.4f}"

                    if 'darwin_best_gen' in res and 'devries_best_gen' in res:
                         darwin_gen_str = res['darwin_best_gen'] if res['darwin_best_gen'] != -1 else 'N/A'
                         devries_gen_str = res['devries_best_gen'] if res['devries_best_gen'] != -1 else 'N/A'
                         line += f",{darwin_gen_str},{devries_gen_str}"

                    f.write(line + "\n")


                f.write("\n")
                f.write("--- Сводная статистика ---\n")
                for key, value in summary_stats.items():
                    translated_key = {
                        "Average Darwin Distance": "Средняя дистанция Дарвин ГА",
                        "Average De Vries Distance": "Средняя дистанция Де Фриз ГА",
                        "Average Exact Distance": "Средняя дистанция Точное",
                        "Average Darwin Time (s)": "Среднее время Дарвин ГА (с)",
                        "Average De Vries Time (s)": "Среднее время Де Фриз ГА (с)",
                        "Average Exact Time (s)": "Среднее время Точное (с)",
                        "Average Darwin Best Gen": "Среднее поколение лучшего Дарвин ГА",
                        "Average De Vries Best Gen": "Среднее поколение лучшего Де Фриз ГА",
                        "Darwin Optimal Found Count": "Дарвин ГА - количество найденных оптимумов",
                        "Darwin Optimal Found Percentage (%)": "Дарвин ГА - процент найденных оптимумов (%)",
                        "De Vries Optimal Found Count": "Де Фриз ГА - количество найденных оптимумов",
                        "De Vries Optimal Percentage (%)": "Де Фриз ГА - процент найденных оптимумов (%)",
                        "Average Darwin Relative Deviation (%)": "Дарвин ГА - среднее отн. отклонение (%)",
                        "Darwin Infinite Relative Deviation Count": "Дарвин ГА - кол-во бесконечных откл.",
                        "Average De Vries Relative Deviation (%)": "Де Фриз ГА - среднее отн. отклонение (%)",
                        "De Vries Infinite Relative Deviation Count": "Де Фриз ГА - кол-во бесконечных откл.",
                    }.get(key, key)

                    if isinstance(value, float):
                         if value == float('inf'):
                             f.write(f"{translated_key}: inf\n")
                         else:
                            f.write(f"{translated_key}: {value:.4f}\n")
                    elif isinstance(value, int) and value == -1:
                         f.write(f"{translated_key}: N/A (не найден)\n")
                    else:
                         f.write(f"{translated_key}: {value}\n")


            self.append_result(f"\nРезультаты пакетного запуска сохранены в файл: {batch_results_filepath}\n")

        except IOError as e:
            messagebox.showerror("Ошибка сохранения пакетных результатов", f"Не удалось сохранить файл пакетных результатов {batch_results_filepath}: {e}")


    def plot_fitness_history(self, darwin_history, devries_history, num_generations, num_cities):
        """Строит график зависимости лучшего фитнеса от поколения."""
        plt.figure(figsize=(10, 6))

        generations_axis = range(num_generations)

        last_darwin_fitness = darwin_history[-1] if darwin_history and darwin_history[-1] != float('inf') else float('inf')
        darwin_history_padded = darwin_history + [last_darwin_fitness] * (num_generations - len(darwin_history))

        last_devries_fitness = devries_history[-1] if devries_history and devries_history[-1] != float('inf') else float('inf')
        devries_history_padded = devries_history + [last_devries_fitness] * (num_generations - len(devries_history))


        plt.plot(generations_axis, darwin_history_padded, label='Дарвин ГА', color='blue')
        plt.plot(generations_axis, devries_history_padded, label='Де Фриз ГА', color='purple')

        plt.xlabel("Поколение")
        plt.ylabel("Лучшая дистанция (фитнес)")
        plt.title(f"История фитнеса ГА ({num_cities} городов, {num_generations} поколений) - Одиночный запуск")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, num_generations - 1)

        all_finite_fitness = [f for f in darwin_history_padded + devries_history_padded if f != float('inf')]
        if all_finite_fitness:
             min_y = min(all_finite_fitness) * 0.95
             max_y = max(all_finite_fitness) * 1.05
             if min_y >= max_y:
                 avg_y = all_finite_fitness[0]
                 min_y = avg_y * 0.9
                 max_y = avg_y * 1.1
             plt.ylim(min_y, max_y)

        plt.show()


    def plot_batch_results(self, all_results, summary_stats):
        """Строит графики сравнения результатов пакетного запуска."""
        if not all_results:
            self.append_result("Нет данных для построения графиков пакетного запуска.\n")
            return

        exact_was_run = self.enable_exact_var.get() and summary_stats.get("Average Exact Distance", float('inf')) != float('inf')

        avg_darwin_time = summary_stats.get("Average Darwin Time (s)", 0.0)
        avg_devries_time = summary_stats.get("Average De Vries Time (s)", 0.0)
        avg_exact_time = summary_stats.get("Average Exact Time (s)", 0.0)

        avg_darwin_relative_deviation = summary_stats.get("Average Darwin Relative Deviation (%)", 0.0)
        avg_devries_relative_deviation = summary_stats.get("Average De Vries Relative Deviation (%)", 0.0)


        num_cities = self.current_num_cities
        iterations = len(all_results)


        plt.figure(figsize=(10, 5))

        # --- График средней ошибки ГА / Средних дистанций ГА ---
        plt.subplot(1, 2, 1)

        if exact_was_run:
             labels_dev = ['Дарвин ГА', 'Де Фриз ГА']
             deviation_values = [avg_darwin_relative_deviation, avg_devries_relative_deviation]

             bars_deviation = plt.bar(labels_dev, deviation_values, color=['blue', 'purple'])

             plt.ylabel("Отклонение (%)")
             plt.title(f"Средняя ошибка ГА относительно точного решения ({iterations} итераций, {num_cities} городов)")
             plt.grid(axis='y', linestyle='--', alpha=0.7)

             for bar in bars_deviation:
                  yval = bar.get_height()
                  if yval != float('inf'):
                     plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', va='bottom', ha='center')
                  else:
                       plt.text(bar.get_x() + bar.get_width()/2, plt.ylim()[1]*0.9, 'inf %', va='top', ha='center', color='red')


             # Удалена строка, добавляющая текст о найденных оптимумах на график
             # darwin_optimal_percent = summary_stats.get("Darwin Optimal Found Percentage (%)", 0)
             # devries_optimal_percent = summary_stats.get("De Vries Optimal Percentage (%)", 0)
             # plt.text(0.5, plt.ylim()[1] * 0.85, f'Оптимум найден: Дарвин {darwin_optimal_percent:.1f}%, Де Фриз {devries_optimal_percent:.1f}%',
             #          ha='center', va='top', fontsize=9, color='dimgray')

             darwin_inf_err = summary_stats.get("Darwin Infinite Relative Deviation Count", 0)
             devries_inf_err = summary_stats.get("De Vries Infinite Relative Deviation Count", 0)
             if darwin_inf_err > 0 or devries_inf_err > 0:
                  plt.text(0.5, plt.ylim()[1] * 0.75, f'Беск. откл. (опт=0, ГА>0): Дарвин {darwin_inf_err}, Де Фриз {devries_inf_err}',
                           ha='center', va='top', fontsize=8, color='darkred')
        else:
             labels_dist = ['Дарвин ГА', 'Де Фриз ГА']
             avg_distances = [summary_stats.get("Average Darwin Distance", float('inf')), summary_stats.get("Average De Vries Distance", float('inf'))]
             avg_distances_finite = [d if d != float('inf') else 0 for d in avg_distances]

             bars_dist = plt.bar(labels_dist, avg_distances_finite, color=['blue', 'purple'])
             plt.ylabel("Средняя дистанция")
             plt.title(f"Средние дистанции ГА ({iterations} итераций, {num_cities} городов) - Точное решение отключено")
             plt.grid(axis='y', linestyle='--', alpha=0.7)

             for i, bar in enumerate(bars_dist):
                  yval = bar.get_height()
                  if avg_distances[i] != float('inf'):
                     plt.text(bar.get_x() + bar.get_width()/2, yval, f'{avg_distances[i]:.2f}', va='bottom', ha='center')
                  else:
                       plt.text(bar.get_x() + bar.get_width()/2, yval, 'inf', va='bottom', ha='center', color='red')


        # --- График сравнения среднего времени ---
        plt.subplot(1, 2, 2)

        labels_time = ['Дарвин ГА', 'Де Фриз ГА']
        avg_times = [avg_darwin_time, avg_devries_time]
        colors_time = ['blue', 'purple']

        if exact_was_run:
             labels_time.append('Точное')
             avg_times.append(avg_exact_time)
             colors_time.append('red')


        bars_time = plt.bar(labels_time, avg_times, color=colors_time)
        plt.ylabel("Среднее время (сек)")
        plt.title(f"Среднее время выполнения ({iterations} итераций, {num_cities} городов)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        all_avg_times = [t for t in avg_times if t is not None and t > 0]
        if len(all_avg_times) > 1 and (max(all_avg_times) / min(all_avg_times) > 100):
             plt.yscale('log')

        for bar in bars_time:
            yval = bar.get_height()
            if yval is not None and yval >= 0:
                 plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', va='bottom', ha='center')


        plt.tight_layout()
        plt.show()


    def start_process(self):
        """Определяет режим работы и запускает соответствующий процесс."""
        if self.distance_matrix is None or self.current_num_cities <= 1:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте или загрузите матрицу расстояний (>1 города).")
            return

        mode = self.mode_var.get()

        if mode == 'batch' and self.enable_exact_var.get() and self.current_num_cities > EXACT_SOLUTION_CITY_LIMIT:
             messagebox.showerror("Ошибка конфигурации", f"Пакетный режим с точным решением возможен только для количества городов <= {EXACT_SOLUTION_CITY_LIMIT}. Пожалуйста, уменьшите количество городов или отключите точное решение.")
             return


        self.run_button.config(state=tk.DISABLED)
        self.num_cities_entry.config(state='readonly')
        self.batch_iterations_entry.config(state='readonly')
        self.enable_exact_checkbox.config(state=tk.DISABLED)

        task_frame_children = self.master.winfo_children()[0].winfo_children()
        try:
            # Отключаем кнопки Generate (1), Load (2), Save (3) в task_frame
            task_frame_children[1].config(state=tk.DISABLED)
            task_frame_children[2].config(state=tk.DISABLED)
            task_frame_children[3].config(state=tk.DISABLED)
        except (IndexError, AttributeError):
            # Используем более устойчивый подход, если индексы могут меняться или виджета нет
            for child in task_frame_children:
                 if isinstance(child, tk.Button) and child['text'] in ["Генерировать новую матрицу", "Загрузить матрицу", "Сохранить текущую матрицу"]:
                      child.config(state=tk.DISABLED)


        try:
            if mode == 'single':
                self._run_single_process()
            elif mode == 'batch':
                 self._run_batch_process()
            else:
                messagebox.showerror("Ошибка режима", "Выбран неизвестный режим работы.")

        except Exception as e:
            messagebox.showerror("Произошла ошибка", f"Ошибка при выполнении: {e}")
            import traceback
            print(traceback.format_exc())
            self.update_status("Произошла ошибка при выполнении.")

        finally:
            self.run_button.config(state=tk.NORMAL)
            self.num_cities_entry.config(state='normal')
            self.batch_iterations_entry.config(state='normal')
            self.enable_exact_checkbox.config(state=tk.NORMAL)

            # Включаем кнопки Generate, Load, Save обратно
            try:
                task_frame_children = self.master.winfo_children()[0].winfo_children()
                for child in task_frame_children:
                     if isinstance(child, tk.Button) and child['text'] in ["Генерировать новую матрицу", "Загрузить матрицу", "Сохранить текущую матрицу"]:
                          child.config(state=tk.NORMAL)
            except (IndexError, AttributeError):
                 print("Предупреждение: Не удалось включить кнопки матрицы.")



    def _run_single_process(self):
        """Выполняет одиночный запуск ГА (Дарвин и Де Фриз) и точного решения."""
        self.clear_results()
        self.append_result(f"Запуск одиночного расчета для {self.current_num_cities} городов...\n")

        darwin_results = {'route': None, 'distance': float('inf'), 'time': 0.0, 'best_generation': -1}
        devries_results = {'route': None, 'distance': float('inf'), 'time': 0.0, 'best_generation': -1}
        exact_results = {'route': None, 'distance': float('inf'), 'time': 0.0}
        darwin_fitness_history = []
        devries_fitness_history = []


        try:
            ga_params = {
                'population_size': self.population_size_var.get(),
                'num_parents': self.num_parents_var.get(),
                'mutation_rate': self.mutation_rate_var.get(),
                'generations': self.generations_var.get()
            }
            if ga_params['population_size'] <= 0 or ga_params['num_parents'] <= 0 or ga_params['num_parents'] > ga_params['population_size'] or \
               not (0.0 <= ga_params['mutation_rate'] <= 1.0) or ga_params['generations'] <= 0:
                 messagebox.showerror("Ошибка ввода", "Некорректные параметры ГА.")
                 return

            num_generations = ga_params['generations']


            # --- Запуск ГА (Дарвин) ---
            self.append_result("\n--- Запуск Генетического алгоритма (Дарвин) ---\n")
            darwin_route, darwin_distance, darwin_time, darwin_fitness_history, darwin_best_gen = run_genetic_algorithm_process(
                ga_params,
                self.distance_matrix,
                model_type='darwin',
                status_callback=self.update_status
            )
            darwin_results = {'route': darwin_route, 'distance': darwin_distance, 'time': darwin_time, 'best_generation': darwin_best_gen}
            self.append_result(f"Лучший найденный маршрут Дарвин ГА: {' -> '.join(map(str, darwin_results['route'])) if darwin_results['route'] else 'N/A'}\n")
            darwin_dist_str = f"{darwin_results['distance']:.4f}" if darwin_results['distance'] != float('inf') else "inf"
            self.append_result(f"Лучшее найденное расстояние Дарвин ГА: {darwin_dist_str}\n")
            self.append_result(f"Время выполнения Дарвин ГА: {darwin_results['time']:.4f} сек.\n")
            if darwin_results['best_generation'] != -1:
                 self.append_result(f"  Найден на поколении: {darwin_results['best_generation']}\n")
            else:
                 self.append_result(f"  Лучшее решение не найдено.\n")


            # --- Запуск ГА (Де Фриз) ---
            self.append_result("\n--- Запуск Генетического алгоритма (Де Фриз) ---\n")
            devries_route, devries_distance, devries_time, devries_fitness_history, devries_best_gen = run_genetic_algorithm_process(
                ga_params,
                self.distance_matrix,
                model_type='devries',
                status_callback=self.update_status
            )
            devries_results = {'route': devries_route, 'distance': devries_distance, 'time': devries_time, 'best_generation': devries_best_gen}
            self.append_result(f"Лучший найденный маршрут Де Фриз ГА: {' -> '.join(map(str, devries_results['route'])) if devries_results['route'] else 'N/A'}\n")
            devries_dist_str = f"{devries_results['distance']:.4f}" if devries_results['distance'] != float('inf') else "inf"
            self.append_result(f"Лучшее найденное расстояние Де Фриз ГА: {devries_dist_str}\n")
            self.append_result(f"Время выполнения Де Фриз ГА: {devries_results['time']:.4f} сек.\n")
            if devries_results['best_generation'] != -1:
                 self.append_result(f"  Найден на поколении: {devries_results['best_generation']}\n")
            else:
                 self.append_result(f"  Лучшее решение не найдено.\n")


            # --- Запуск поиска точного решения (условно) ---
            self.append_result("\n--- Поиск точного решения ---\n")

            if self.enable_exact_var.get():
                 if self.current_num_cities > EXACT_SOLUTION_CITY_LIMIT:
                      self.append_result(f"Пропуск поиска точного решения перебором: слишком много городов ({self.current_num_cities} > {EXACT_SOLUTION_CITY_LIMIT}).\n")
                      self.update_status("Расчеты завершены (Точное решение пропущено из-за лимита).")
                 else:
                      exact_route, exact_distance, exact_time = find_exact_solution_brute_force(
                          self.distance_matrix,
                          status_callback=self.update_status
                      )
                      exact_results = {'route': exact_route, 'distance': exact_distance, 'time': exact_time}
                      self.append_result(f"Оптимальный маршрут: {' -> '.join(map(str, exact_results['route'])) if exact_results['route'] else 'N/A'}\n")
                      exact_dist_str = f"{exact_results['distance']:.4f}" if exact_results['distance'] != float('inf') else "inf"
                      self.append_result(f"Минимальная дистанция: {exact_dist_str}\n")
                      self.append_result(f"Время выполнения точного решения: {exact_results['time']:.4f} сек.\n")
                      self.update_status("Расчеты завершены.")
            else:
                 self.append_result("Поиск точного решения отключен пользователем.\n")
                 self.update_status("Расчеты завершены (Точное решение отключено).")


            # --- Сводка сравнения ---
            self.append_result("\n--- Сводка сравнения (одиночный) ---\n")
            methods = {'Дарвин ГА': darwin_results, 'Де Фриз ГА': devries_results}
            if exact_results['route'] is not None:
                methods['Точное'] = exact_results

            min_dist = float('inf')
            best_method = 'N/A'

            found_solutions = {name: res for name, res in methods.items() if res['distance'] != float('inf')}
            if found_solutions:
                min_dist = min(found_solutions.values(), key=lambda x: x['distance'])['distance']
                best_method = min(found_solutions.items(), key=lambda item: item[1]['distance'])[0]
                self.append_result(f"Лучшее из найденных: {min_dist:.4f} ({best_method})\n")
            else:
                 self.append_result("Нет найденных решений для сравнения.\n")


            if exact_results['distance'] != float('inf'):
                 self.append_result(f"\n--- Отклонение от точного решения ---\n")
                 for name, res in methods.items():
                      if res['distance'] != float('inf'):
                           abs_dev = abs(res['distance'] - exact_results['distance'])
                           self.append_result(f"Дистанция {name}: {res['distance']:.4f} (Отклонение от точного: {abs_dev:.4f})\n")
                           if exact_results['distance'] > 0:
                                rel_dev = (abs_dev / exact_results['distance']) * 100.0
                                self.append_result(f"  Отн. откл. от точного: {rel_dev:.2f} %\n")
                           elif abs_dev > 0:
                                self.append_result(f"  Отн. откл. от точного: inf %\n")
                           else:
                                 self.append_result(f"  Отн. откл. от точного: 0.0 %\n")

                      elif name in ['Дарвин ГА', 'Де Фриз ГА']:
                            self.append_result(f"Дистанция {name}: N/A (Не найден корректный маршрут)\n")


            elif exact_results['route'] is not None:
                 self.append_result("\nТочное решение было запущено, но не дало конечного результата.\n")
            else:
                  self.append_result("\nТочное решение было отключено или пропущено, сравнение отклонений невозможно.\n")


            self.append_result(f"\nВремя выполнения: Дарвин ГА {darwin_results['time']:.4f} сек., Де Фриз ГА {devries_results['time']:.4f} сек.")
            if exact_results['route'] is not None or exact_results['time'] > 0:
                 self.append_result(f", Точное {exact_results['time']:.4f} сек.\n")
            else:
                 self.append_result("\n")


            self.save_single_results_to_file(darwin_results, devries_results, exact_results)

            if self.current_num_cities > 1 and num_generations > 0:
                 self.plot_fitness_history(
                     darwin_fitness_history,
                     devries_fitness_history,
                     num_generations,
                     self.current_num_cities
                 )
            else:
                 self.append_result("\nГрафик истории фитнеса не построен (слишком мало городов или поколений).\n")


        except ValueError:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите числовые значения во все поля параметров.")
            self.update_status("Ошибка ввода параметров.")
        except Exception as e:
            messagebox.showerror("Произошла ошибка", f"Общая ошибка при выполнении алгоритмов: {e}")
            import traceback
            print(traceback.format_exc())
            self.update_status("Произошла ошибка при выполнении.")
            raise


    def _run_batch_process(self):
        """Выполняет пакетный запуск ГА (Дарвин и Де Фриз) и точного решения N раз."""
        try:
            num_iterations_str = self.batch_iterations_var.get()
            try:
                num_iterations = int(num_iterations_str)
                if num_iterations <= 0:
                     messagebox.showerror("Ошибка ввода", "Количество итераций для пакетного режима должно быть > 0.")
                     return
            except ValueError:
                 messagebox.showerror("Ошибка ввода", "Пожалуйста, введите целое число для количества итераций.")
                 return

            self.clear_results()
            self.append_result(f"Запуск пакетного режима ({num_iterations} итераций) для {self.current_num_cities} городов...\n")

            all_results = []

            ga_params = {
                'population_size': self.population_size_var.get(),
                'num_parents': self.num_parents_var.get(),
                'mutation_rate': self.mutation_rate_var.get(),
                'generations': self.generations_var.get()
            }
            if ga_params['population_size'] <= 0 or ga_params['num_parents'] <= 0 or ga_params['num_parents'] > ga_params['population_size'] or \
               not (0.0 <= ga_params['mutation_rate'] <= 1.0) or ga_params['generations'] <= 0:
                 messagebox.showerror("Ошибка ввода", "Некорректные параметры ГА для пакетного режима.")
                 return


            for i in range(num_iterations):
                self.update_status(f"Пакетный режим: Итерация {i + 1}/{num_iterations}")
                current_matrix = generate_random_distance_matrix(self.current_num_cities)

                darwin_route, darwin_distance, darwin_time, _, darwin_best_gen = run_genetic_algorithm_process(
                    ga_params,
                    current_matrix,
                    model_type='darwin'
                )

                devries_route, devries_distance, devries_time, _, devries_best_gen = run_genetic_algorithm_process(
                    ga_params,
                    current_matrix,
                    model_type='devries'
                )

                exact_distance = float('inf')
                exact_time = 0.0
                if self.enable_exact_var.get():
                    exact_route, exact_distance, exact_time = find_exact_solution_brute_force(
                        current_matrix
                    )

                all_results.append({
                    'iter': i + 1,
                    'darwin_distance': darwin_distance,
                    'darwin_time': darwin_time,
                    'darwin_best_gen': darwin_best_gen,
                    'devries_distance': devries_distance,
                    'devries_time': devries_time,
                    'devries_best_gen': devries_best_gen,
                    'exact_distance': exact_distance,
                    'exact_time': exact_time
                })


            self.update_status("Пакетный режим: Все итерации завершены. Обработка результатов...")
            self.append_result("\n--- Результаты пакетного режима ---\n")

            # Расчет сводной статистики
            darwin_distances = [r['darwin_distance'] for r in all_results if r['darwin_distance'] != float('inf')]
            devries_distances = [r['devries_distance'] for r in all_results if r['devries_distance'] != float('inf')]
            valid_exact_distances = [r['exact_distance'] for r in all_results if r['exact_distance'] != float('inf')]


            summary_stats = {}
            summary_stats["Average Darwin Distance"] = sum(darwin_distances) / len(darwin_distances) if darwin_distances else float('inf')
            summary_stats["Average De Vries Distance"] = sum(devries_distances) / len(devries_distances) if devries_distances else float('inf')
            summary_stats["Average Exact Distance"] = sum(valid_exact_distances) / len(valid_exact_distances) if valid_exact_distances else float('inf')

            summary_stats["Average Darwin Time (s)"] = sum(r['darwin_time'] for r in all_results) / num_iterations
            summary_stats["Average De Vries Time (s)"] = sum(r['devries_time'] for r in all_results) / num_iterations
            summary_stats["Average Exact Time (s)"] = sum(r['exact_time'] for r in all_results) / num_iterations

            valid_darwin_gens = [r['darwin_best_gen'] for r in all_results if r['darwin_best_gen'] != -1]
            valid_devries_gens = [r['devries_best_gen'] for r in all_results if r['devries_best_gen'] != -1]

            summary_stats["Average Darwin Best Gen"] = sum(valid_darwin_gens) / len(valid_darwin_gens) if valid_darwin_gens else -1
            summary_stats["Average De Vries Best Gen"] = sum(valid_devries_gens) / len(valid_devries_gens) if valid_devries_gens else -1


            darwin_optimal_count = sum(1 for r in all_results if r['darwin_distance'] != float('inf') and r['exact_distance'] != float('inf') and abs(r['darwin_distance'] - r['exact_distance']) < 1e-6)
            summary_stats["Darwin Optimal Found Count"] = darwin_optimal_count
            summary_stats["Darwin Optimal Found Percentage (%)"] = (darwin_optimal_count / num_iterations) * 100.0 if num_iterations > 0 else 0

            devries_optimal_count = sum(1 for r in all_results if r['devries_distance'] != float('inf') and r['exact_distance'] != float('inf') and abs(r['devries_distance'] - r['exact_distance']) < 1e-6)
            summary_stats["De Vries Optimal Found Count"] = devries_optimal_count
            summary_stats["De Vries Optimal Percentage (%)"] = (devries_optimal_count / num_iterations) * 100.0 if num_iterations > 0 else 0

            darwin_relative_deviations = []
            devries_relative_deviations = []

            for r in all_results:
                if r['exact_distance'] != float('inf'):
                    if r['darwin_distance'] != float('inf'):
                         abs_dev_d = abs(r['darwin_distance'] - r['exact_distance'])
                         if r['exact_distance'] > 0:
                              darwin_relative_deviations.append((abs_dev_d / r['exact_distance']) * 100.0)
                         elif abs_dev_d > 0:
                              darwin_relative_deviations.append(float('inf'))
                         else:
                             darwin_relative_deviations.append(0.0)

                    if r['devries_distance'] != float('inf'):
                         abs_dev_dv = abs(r['devries_distance'] - r['exact_distance'])
                         if r['exact_distance'] > 0:
                              devries_relative_deviations.append((abs_dev_dv / r['exact_distance']) * 100.0)
                         elif abs_dev_dv > 0:
                              devries_relative_deviations.append(float('inf'))
                         else:
                             devries_relative_deviations.append(0.0)

            finite_darwin_rel_dev = [d for d in darwin_relative_deviations if d != float('inf')]
            summary_stats["Average Darwin Relative Deviation (%)"] = sum(finite_darwin_rel_dev) / len(finite_darwin_rel_dev) if finite_darwin_rel_dev else 0.0
            summary_stats["Darwin Infinite Relative Deviation Count"] = sum(1 for d in darwin_relative_deviations if d == float('inf'))

            finite_devries_rel_dev = [d for d in devries_relative_deviations if d != float('inf')]
            summary_stats["Average De Vries Relative Deviation (%)"] = sum(finite_devries_rel_dev) / len(finite_devries_rel_dev) if finite_devries_rel_dev else 0.0
            summary_stats["De Vries Infinite Relative Deviation Count"] = sum(1 for d in devries_relative_deviations if d == float('inf'))


            for key, value in summary_stats.items():
                translated_key = {
                        "Average Darwin Distance": "Средняя дистанция Дарвин ГА",
                        "Average De Vries Distance": "Средняя дистанция Де Фриз ГА",
                        "Average Exact Distance": "Средняя дистанция Точное",
                        "Average Darwin Time (s)": "Среднее время Дарвин ГА (с)",
                        "Average De Vries Time (s)": "Среднее время Де Фриз ГА (с)",
                        "Average Exact Time (s)": "Среднее время Точное (с)",
                        "Average Darwin Best Gen": "Среднее поколение лучшего Дарвин ГА",
                        "Average De Vries Best Gen": "Среднее поколение лучшего Де Фриз ГА",
                        "Darwin Optimal Found Count": "Дарвин ГА - количество найденных оптимумов",
                        "Darwin Optimal Found Percentage (%)": "Дарвин ГА - процент найденных оптимумов (%)",
                        "De Vries Optimal Found Count": "Де Фриз ГА - количество найденных оптимумов",
                        "De Vries Optimal Percentage (%)": "Де Фриз ГА - процент найденных оптимумов (%)",
                        "Average Darwin Relative Deviation (%)": "Дарвин ГА - среднее отн. отклонение (%)",
                        "Darwin Infinite Relative Deviation Count": "Дарвин ГА - кол-во бесконечных откл.",
                        "Average De Vries Relative Deviation (%)": "Де Фриз ГА - среднее отн. отклонение (%)",
                        "De Vries Infinite Relative Deviation Count": "Де Фриз ГА - кол-во бесконечных откл.",
                    }.get(key, key)

                if isinstance(value, float):
                     if value == float('inf'):
                         self.append_result(f"{translated_key}: inf\n")
                     else:
                         self.append_result(f"{translated_key}: {value:.4f}\n")
                elif isinstance(value, int) and value == -1:
                     self.append_result(f"{translated_key}: N/A (не найден)\n")
                else:
                     self.append_result(f"{translated_key}: {value}\n")

            self.save_batch_results_to_file(all_results, summary_stats)

            self.plot_batch_results(all_results, summary_stats)


            self.update_status("Пакетный режим завершен. Графики показаны.")


        except ValueError:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите числовые значения в поля параметров пакетного режима.")
            self.update_status("Ошибка ввода параметров пакетного режима.")
        except Exception as e:
            messagebox.showerror("Произошла ошибка", f"Общая ошибка при выполнении пакетного режима: {e}")
            import traceback
            print(traceback.format_exc())
            self.update_status("Произошла ошибка при выполнении пакетного режима.")
            raise

        finally:
            pass


# --- Запуск GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TSPSolverGUI(root)
    root.mainloop()
import matplotlib.pyplot as plt
import random
import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

def read_routes_from_results(filepath):
    """
    Читает файл результатов сравнения и извлекает маршруты.
    Возвращает словарь {'ga_route': [...], 'exact_route': [...]}.
    """
    routes = {'ga_route': None, 'exact_route': None}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                match_ga_route = re.match(r"GA Best Route:\s*([\d\s]+)", line)
                match_exact_route = re.match(r"Exact Best Route:\s*([\d\s]+)", line)

                if match_ga_route:
                    try:
                        routes['ga_route'] = [int(city) for city in match_ga_route.group(1).split()]
                    except ValueError:
                        messagebox.showwarning("Ошибка парсинга", f"Не удалось разобрать GA маршрут: {line}")
                elif match_exact_route:
                    try:
                         route_str = match_exact_route.group(1).strip() # Добавил strip()
                         if route_str.upper() != "N/A":
                            routes['exact_route'] = [int(city) for city in route_str.split()]
                         else:
                             routes['exact_route'] = None # Если в файле "N/A", сохраняем как None
                    except ValueError:
                        messagebox.showwarning("Ошибка парсинга", f"Не удалось разобрать точный маршрут: {line}")


        if not routes['ga_route'] and not routes['exact_route']:
             messagebox.showwarning("Неполные данные", f"В файле {filepath} не найдено ни одного корректного маршрута.")
             return None

        return routes

    except FileNotFoundError:
        messagebox.showerror("Ошибка файла", f"Файл не найден: {filepath}")
        return None
    except Exception as e:
        messagebox.showerror("Произошла ошибка", f"Ошибка при чтении файла {filepath}: {e}")
        return None

def draw_route(route, title, city_coords):
    """
    Рисует маршрут на графике и отображает номера городов.
    city_coords - словарь {city_number: (x, y)}
    """
    if not route or not city_coords:
        print("Нет данных для рисования.")
        return

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Рисуем города и их номера
    for city, (x, y) in city_coords.items():
        color = 'red' if city == 1 else 'blue'
        plt.plot(x, y, 'o', markersize=8, color=color) # Город 1 красным, остальные синим
        # Отображаем номер города рядом с точкой
        plt.text(x, y, str(city), fontsize=10, ha='right', va='bottom') # Можно настроить ha/va для положения текста


    # Рисуем маршрут
    # Получаем координаты в порядке маршрута
    valid_route_coords = []
    for city in route:
        if city in city_coords:
            valid_route_coords.append(city_coords[city])
        else:
             print(f"Предупреждение: Для города {city} нет координат.")


    if len(valid_route_coords) < len(route):
         messagebox.showwarning("Неполное рисование", "Не удалось найти координаты для всех городов в маршруте. Маршрут может быть нарисован не полностью.")


    # Рисуем линии между городами в маршруте
    for i in range(len(valid_route_coords) - 1):
        x1, y1 = valid_route_coords[i]
        x2, y2 = valid_route_coords[i+1]
        plt.plot([x1, x2], [y1, y2], 'gray', linestyle='-', linewidth=1) # Серая линия

    # Дополнительно выделяем первую и последнюю часть маршрута
    if len(valid_route_coords) > 1:
         plt.plot([valid_route_coords[0][0], valid_route_coords[1][0]], [valid_route_coords[0][1], valid_route_coords[1][1]], 'green', linestyle='-', linewidth=2, label='Начало маршрута') # Начало зеленым
    if len(valid_route_coords) > 1 and valid_route_coords[-1] == valid_route_coords[0]: # Рисуем конечный сегмент, только если маршрут замкнут и есть хотя бы 2 уникальных города
         plt.plot([valid_route_coords[-2][0], valid_route_coords[-1][0]], [valid_route_coords[-2][1], valid_route_coords[-1][1]], 'purple', linestyle='-', linewidth=2, label='Конец маршрута (возврат)') # Конец фиолетовым


    plt.title(title)
    plt.xlabel("X Координата")
    plt.ylabel("Y Координата")
    plt.grid(True)
    plt.axis('equal') # Сохраняем пропорции осей
    # Добавляем легенду
    plt.legend()
    plt.show()

# --- Запуск скрипта ---
if __name__ == "__main__":
    # Создаем корневое окно Tkinter, но не показываем его
    root = tk.Tk()
    root.withdraw()

    # Открываем диалог выбора файла
    filepath = filedialog.askopenfilename(
        title="Выберите файл с результатами сравнения (*_results.txt)",
        filetypes=(("Файлы результатов TSP", "*_results.txt"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
    )

    # Если файл выбран
    if filepath:
        routes = read_routes_from_results(filepath)

        if routes:
            available_routes = []
            if routes.get('ga_route'):
                 available_routes.append("1: Маршрут Генетического алгоритма")
            if routes.get('exact_route'):
                 available_routes.append("2: Точный маршрут")

            if not available_routes:
                 messagebox.showinfo("Нет маршрутов", "В выбранном файле не найдено корректных маршрутов для отрисовки.")
            else:
                 # Спрашиваем пользователя, какой маршрут нарисовать
                 choice = simpledialog.askstring("Выбор маршрута",
                                                 "Какой маршрут вы хотите нарисовать?\n" + "\n".join(available_routes),
                                                 parent=root)

                 selected_route = None
                 plot_title = "Маршрут TSP"

                 if choice == '1' and routes.get('ga_route') is not None: # Проверяем, что маршрут не None
                      selected_route = routes['ga_route']
                      plot_title = "Маршрут Генетического алгоритма"
                 elif choice == '2' and routes.get('exact_route') is not None: # Проверяем, что маршрут не None
                      selected_route = routes['exact_route']
                      plot_title = "Точный маршрут"
                 else:
                      messagebox.showwarning("Некорректный выбор", "Выбран некорректный пункт или отмена.")


                 if selected_route:
                    # Определяем уникальные города в выбранном маршруте
                    # Исключаем первый и последний город (они оба равны 1)
                    # Используем set, чтобы получить только уникальные номера
                    cities_in_route_set = set(selected_route)
                    # Убедимся, что город 1 присутствует
                    if 1 not in cities_in_route_set and selected_route: # Добавляем 1, если он должен быть
                         cities_in_route_set.add(1)

                    cities_in_route_list = sorted(list(cities_in_route_set))
                    num_unique_cities = len(cities_in_route_list)


                    # Генерируем случайные координаты для каждого уникального города в маршруте
                    city_coords = {}
                    for city_num in cities_in_route_list:
                         city_coords[city_num] = (random.uniform(0, 100), random.uniform(0, 100))

                    # Убедимся, что у нас есть координаты для всех городов, упомянутых в выбранном маршруте
                    if all(city in city_coords for city in selected_route):
                         draw_route(selected_route, plot_title, city_coords)
                    else:
                         # Это unlikely должно произойти после добавления всех уникальных городов
                         messagebox.showerror("Ошибка", "Внутренняя ошибка: Не удалось сгенерировать координаты для всех городов в выбранном маршруте.")

    # Уничтожаем корневое окно Tkinter
    root.destroy()
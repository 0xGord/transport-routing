import os
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Tuple, List


def read_vrp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = {}
    node_coords = []
    demands = []
    distance_matrix = []
    
    node_coord_section = False
    demand_section = False
    edge_weight_section = False
    dimension = 0
    edge_weight_type = "EUC_2D"  # по умолчанию евклидово расстояние
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("DIMENSION"):
            dimension = int(line.split()[-1])
            data['dimension'] = dimension
        elif line.startswith("CAPACITY"):
            data['capacity'] = int(line.split()[-1])
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split()[-1]
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            edge_weight_section = True
            node_coord_section = False
            demand_section = False
        elif line.startswith("NODE_COORD_SECTION"):
            node_coord_section = True
            demand_section = False
            edge_weight_section = False
        elif line.startswith("DEMAND_SECTION"):
            demand_section = True
            node_coord_section = False
            edge_weight_section = False
        elif line.startswith("DEPOT_SECTION"):
            break
        elif edge_weight_section:
            # Читаем матрицу расстояний
            numbers = line.split()
            for num in numbers:
                if num:
                    distance_matrix.append(float(num))
        elif node_coord_section:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                node_coords.append((node_id, float(parts[1]), float(parts[2])))
        elif demand_section:
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands.append((node_id, demand))
    
    # Проверяем, что dimension определен
    if dimension == 0:
        # Пытаемся определить по максимальному ID
        if node_coords:
            dimension = max([c[0] for c in node_coords])
        elif demands:
            dimension = max([d[0] for d in demands])
        else:
            dimension = 0
    
    # Для формата с явной матрицей расстояний
    if edge_weight_type == "EXPLICIT" and distance_matrix:
        data['edge_weight_type'] = edge_weight_type
        data['distance_matrix'] = distance_matrix
        # Создаем фиктивные координаты для совместимости
        for i in range(1, dimension + 1):
            node_coords.append((i, 0.0, 0.0))
    
    # Сортируем по ID и создаем списки с индексацией с 0
    node_coords.sort(key=lambda x: x[0])
    demands.sort(key=lambda x: x[0])
    
    # Создаем списки в правильном порядке (индексация с 0)
    formatted_coords = []
    formatted_demands = []
    
    for i, (node_id, x, y) in enumerate(node_coords):
        formatted_coords.append((i, x, y))
    
    for i, (node_id, demand) in enumerate(demands):
        formatted_demands.append((i, demand))
    
    data['node_coords'] = formatted_coords
    data['demands'] = formatted_demands
    data['dimension'] = len(formatted_coords)
    
    return data


def read_sol(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    routes = []
    cost = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith("Route"):
            parts = line.split(":")[1].strip().split()
            # Конвертируем ID узлов из файла (начиная с 1) в наши индексы (начиная с 0)
            route = [int(node) for node in parts]
            # Смещаем все узлы на -1, так как в файле нумерация с 1, а в алгоритме с 0
            route = [node - 1 for node in route]
            routes.append(route)
        elif line.startswith("Cost"):
            cost = int(line.split()[-1])
    
    return routes, cost


def calculate_distance(coord1: List[float], coord2: List[float]) -> float:
    distance = math.sqrt((coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
    return distance if distance != 0 else 1e-10


def initialize_pheromones(dimension: int, initial_pheromone: float) -> np.ndarray:
    return np.full((dimension, dimension), initial_pheromone)


def calculate_probabilities(
    pheromones: np.ndarray, 
    distances: np.ndarray, 
    visited: List[int], 
    alpha: float, 
    beta: float
) -> np.ndarray:
    probabilities = np.zeros(len(distances))
    current = visited[-1]
    for i in range(len(distances)):
        if i not in visited:
            if distances[current, i] == 0:
                probabilities[i] = 0
            else:
                probabilities[i] = (pheromones[current, i] ** alpha) * ((1.0 / distances[current, i]) ** beta)

    if probabilities.sum() == 0:
        probabilities = np.ones(len(distances)) / len(distances)
    else:
        probabilities /= probabilities.sum()
    
    return probabilities


def ant_colony_optimization(
    data: Dict[str, Any], 
    num_ants: int, 
    num_iterations: int, 
    alpha: float, 
    beta: float, 
    evaporation_rate: float, 
    initial_pheromone: float
) -> Tuple[List[int], float]:
    dimension = data['dimension']
    capacity = data['capacity']
    node_coords = data['node_coords']
    demands = data['demands']
    
    # Проверяем, что данные загружены корректно
    if dimension == 0 or len(node_coords) == 0:
        print(f"  Ошибка: нет данных для задачи (dimension={dimension}, nodes={len(node_coords)})")
        return [], float('inf')
    
    # Создаем матрицу расстояний
    distances = np.zeros((dimension, dimension))
    
    # Если есть явная матрица расстояний, используем её
    if 'distance_matrix' in data and data.get('edge_weight_type') == 'EXPLICIT':
        dm = data['distance_matrix']
        idx = 0
        # Матрица в формате LOWER_ROW (только нижний треугольник без диагонали)
        for i in range(1, dimension):
            for j in range(i):
                if idx < len(dm):
                    dist = dm[idx]
                    distances[i, j] = dist
                    distances[j, i] = dist
                    idx += 1
        # Диагональ (расстояние от узла до самого себя)
        for i in range(dimension):
            distances[i, i] = 0
    else:
        # Иначе вычисляем евклидово расстояние
        for i in range(dimension):
            for j in range(dimension):
                if i < len(node_coords) and j < len(node_coords):
                    distances[i, j] = calculate_distance(node_coords[i], node_coords[j])
                else:
                    distances[i, j] = 1e10  # Большое расстояние для отсутствующих узлов
    
    pheromones = initialize_pheromones(dimension, initial_pheromone)
    
    best_cost = float('inf')
    best_routes = []
    
    for iteration in range(num_iterations):
        all_routes = []
        all_costs = []
        
        for ant in range(num_ants):
            current_node = 0  # Начинаем с депо
            visited = [current_node]
            remaining_capacity = capacity
            route = [current_node]
            cost = 0
            
            while len(visited) < dimension:
                probabilities = calculate_probabilities(pheromones, distances, visited, alpha, beta)
                
                # Выбираем следующий узел
                try:
                    next_node = np.random.choice(range(dimension), p=probabilities)
                except ValueError:
                    # Если probabilities содержит NaN или другие ошибки, выбираем случайно
                    available = [i for i in range(dimension) if i not in visited]
                    if available:
                        next_node = np.random.choice(available)
                    else:
                        break
                
                # Получаем спрос для следующего узла
                if next_node < len(demands):
                    demand = demands[next_node][1]
                else:
                    demand = 0
                
                # Проверяем, можно ли посетить узел
                if next_node not in visited and remaining_capacity >= demand and remaining_capacity > 0:
                    visited.append(next_node)
                    route.append(next_node)
                    remaining_capacity -= demand
                    cost += distances[current_node, next_node]
                    current_node = next_node
                else:
                    # Возвращаемся в депо
                    cost += distances[current_node, 0]
                    route.append(0)
                    remaining_capacity = capacity
                    current_node = 0
            
            # Возвращаемся в депо в конце
            cost += distances[current_node, 0]
            route.append(0)
            all_routes.append(route)
            all_costs.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_routes = route
        
        # Обновляем феромоны
        pheromones *= (1 - evaporation_rate)
        for route, cost in zip(all_routes, all_costs):
            if cost > 0:
                for i in range(len(route) - 1):
                    if route[i] < dimension and route[i+1] < dimension:
                        pheromones[route[i], route[i+1]] += 1.0 / cost
    
    return best_routes, best_cost


def calculate_deviation(cost: float, optimal_cost: float) -> float:
    return ((cost - optimal_cost) / optimal_cost) * 100


def get_result_filename(results_dir: str, set_name: str, file_name: str) -> str:
    """Возвращает имя файла для сохранения результата конкретной задачи"""
    base_name = file_name.replace('.vrp', '')
    return os.path.join(results_dir, 'calculations', f"{set_name}_{base_name}_result.json")


def is_already_calculated(results_dir: str, set_name: str, file_name: str) -> bool:
    """Проверяет, был ли уже рассчитан файл"""
    result_file = get_result_filename(results_dir, set_name, file_name)
    return os.path.exists(result_file)


def convert_to_serializable(obj):
    """Конвертирует numpy типы в стандартные Python типы для JSON сериализации"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_calculation_result(results_dir: str, set_name: str, file_name: str, 
                           data: Dict, best_routes: List[int], best_cost: float, 
                           optimal_cost: float, deviation: float, execution_time: float):
    """Сохраняет результат расчета конкретной задачи"""
    result_file = get_result_filename(results_dir, set_name, file_name)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    # Конвертируем все numpy типы в стандартные Python типы
    result = {
        'file_name': file_name,
        'set_name': set_name,
        'dimension': int(data['dimension']),
        'optimal_cost': int(optimal_cost),
        'best_cost': float(best_cost),
        'deviation': float(deviation),
        'execution_time': float(execution_time),
        'best_routes': [int(node) for node in best_routes],
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'num_ants': int(num_ants),
            'num_iterations': int(num_iterations),
            'alpha': float(alpha),
            'beta': float(beta),
            'evaporation_rate': float(evaporation_rate),
            'initial_pheromone': float(initial_pheromone)
        }
    }
    
    # Дополнительная конвертация для безопасности
    result = convert_to_serializable(result)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def load_calculation_result(results_dir: str, set_name: str, file_name: str) -> Dict:
    """Загружает ранее сохраненный результат"""
    result_file = get_result_filename(results_dir, set_name, file_name)
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results_to_file(results_dir: str, set_name: str, results_data: str):
    """Сохраняет результаты в текстовый файл"""
    filename = os.path.join(results_dir, f"{set_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(results_data)
    return filename


def process_folder_with_aco(
    folder_path: str, 
    num_ants: int, 
    num_iterations: int, 
    alpha: float, 
    beta: float, 
    evaporation_rate: float, 
    initial_pheromone: float,
    results_dir: str,
    results_buffer: List[str],
    force_recalculate: bool = False
) -> Tuple[List[float], List[float], List[int]]:   
    deviations = []
    execution_times = []
    dimensions = []
    
    set_name = os.path.basename(folder_path)
    results_buffer.append(f"\n{'='*60}")
    results_buffer.append(f"Обработка набора {set_name}")
    results_buffer.append('='*60)
    
    # Получаем список всех .vrp файлов
    vrp_files = [f for f in os.listdir(folder_path) if f.endswith(".vrp")]
    vrp_files.sort()  # Сортируем для консистентности
    
    for file_name in vrp_files:
        vrp_file_path = os.path.join(folder_path, file_name)
        sol_file_path = os.path.join(folder_path, file_name.replace(".vrp", ".sol"))
        
        if not os.path.exists(sol_file_path):
            msg = f"Файл {sol_file_path} не найден, пропускаем {file_name}."
            print(msg)
            results_buffer.append(msg)
            continue
        
        # Проверяем, был ли файл уже рассчитан
        if not force_recalculate and is_already_calculated(results_dir, set_name, file_name):
            try:
                result = load_calculation_result(results_dir, set_name, file_name)
                deviations.append(result['deviation'])
                execution_times.append(result['execution_time'])
                dimensions.append(result['dimension'])
                
                output = []
                output.append(f"\nФайл: {file_name} (загружено из кэша)")
                output.append(f"Размерность задачи: {result['dimension']}")
                output.append(f"Тип задачи: {set_name}")
                output.append(f"Оптимальная стоимость: {result['optimal_cost']}")
                output.append(f"Стоимость вашего решения: {result['best_cost']:.2f}")
                output.append(f"Отклонение от оптимального решения: {result['deviation']:.2f}%")
                output.append(f"Время выполнения: {result['execution_time']:.2f} секунд")
                output.append("-" * 40)
                
                for line in output:
                    print(line)
                    results_buffer.append(line)
                
                continue
            except Exception as e:
                msg = f"Ошибка загрузки кэша для {file_name}: {e}. Пересчитываем..."
                print(msg)
                results_buffer.append(msg)
        
        # Рассчитываем файл
        print(f"  Обработка {file_name}...")
        data = read_vrp(vrp_file_path)
        
        # Проверяем, что данные загружены корректно
        if data['dimension'] == 0 or len(data['node_coords']) == 0:
            msg = f"  Ошибка: не удалось загрузить данные из {file_name}"
            print(msg)
            results_buffer.append(msg)
            continue
        
        optimal_routes, optimal_cost = read_sol(sol_file_path)
        
        start_time = time.time()
        best_routes, best_cost = ant_colony_optimization(data, num_ants, num_iterations, alpha, beta, evaporation_rate, initial_pheromone)
        execution_time = time.time() - start_time
        
        if best_cost == float('inf'):
            msg = f"  Ошибка: не удалось найти решение для {file_name}"
            print(msg)
            results_buffer.append(msg)
            continue
        
        deviation = calculate_deviation(best_cost, optimal_cost)
        deviations.append(deviation)
        execution_times.append(execution_time)
        dimensions.append(data['dimension'])
        
        # Сохраняем результат
        save_calculation_result(results_dir, set_name, file_name, data, 
                               best_routes, best_cost, optimal_cost, deviation, execution_time)
        
        # Формируем вывод
        output = []
        output.append(f"\nФайл: {file_name}")
        output.append(f"Размерность задачи: {data['dimension']}")
        output.append(f"Тип задачи: {set_name}")
        output.append(f"Оптимальная стоимость: {optimal_cost}")
        output.append(f"Стоимость вашего решения: {best_cost:.2f}")
        output.append(f"Отклонение от оптимального решения: {deviation:.2f}%")
        output.append(f"Время выполнения: {execution_time:.2f} секунд")
        output.append("\nОптимальный маршрут:")
        for route in optimal_routes:
            # Конвертируем обратно в нумерацию с 1 для вывода
            route_display = [node + 1 for node in route]
            output.append(" -> ".join(map(str, route_display)))
        output.append("\nНайденный маршрут:")
        # Разбиваем найденный маршрут на отдельные маршруты по возвратам в депо (0)
        routes = []
        current_route = []
        for node in best_routes:
            if node == 0:
                if current_route:
                    routes.append(current_route)
                current_route = [1]  # Депо выводим как 1
            else:
                current_route.append(node + 1)  # Конвертируем в нумерацию с 1
        if current_route and current_route != [1]:
            routes.append(current_route)
        
        for route in routes:
            output.append(" -> ".join(map(str, route)))
        output.append("-" * 40)
        
        # Выводим в консоль и сохраняем в буфер
        for line in output:
            print(line)
            results_buffer.append(line)
    
    return deviations, execution_times, dimensions


def print_set_statistics(set_name, deviations, times, dimensions, results_buffer):
    """Выводит статистику по набору и сохраняет в буфер"""
    if deviations:
        avg_dev = sum(deviations) / len(deviations)
        avg_time = sum(times) / len(times)
        
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Статистика для набора {set_name}:")
        output.append(f"Среднее отклонение: {avg_dev:.2f}%")
        output.append(f"Среднее время выполнения: {avg_time:.2f} секунд")
        output.append('='*60)
        
        # Группируем по размерностям
        unique_dims = sorted(list(set(dimensions)))
        for dim in unique_dims:
            indices = [i for i, x in enumerate(dimensions) if x == dim]
            if indices:
                avg_dev_dim = sum([deviations[i] for i in indices]) / len(indices)
                avg_time_dim = sum([times[i] for i in indices]) / len(indices)
                output.append(f"  Размерность {dim}: отклонение = {avg_dev_dim:.2f}%, время = {avg_time_dim:.2f} сек")
        
        for line in output:
            print(line)
            results_buffer.append(line)
    else:
        msg = f"\nНет данных для набора {set_name}"
        print(msg)
        results_buffer.append(msg)


def aggregate_by_dimension(dimensions, deviations, times):
    """Агрегирует данные по уникальным размерностям (берет среднее)"""
    if not dimensions:
        return [], [], []
    
    unique_dims = sorted(list(set(dimensions)))
    agg_deviations = []
    agg_times = []
    for dim in unique_dims:
        indices = [i for i, x in enumerate(dimensions) if x == dim]
        if indices:
            agg_deviations.append(sum([deviations[i] for i in indices]) / len(indices))
            agg_times.append(sum([times[i] for i in indices]) / len(indices))
        else:
            agg_deviations.append(0)
            agg_times.append(0)
    return unique_dims, agg_deviations, agg_times


# Параметры алгоритма
num_ants = 50
num_iterations = 500
alpha = 1
beta = 2
evaporation_rate = 0.5
initial_pheromone = 1.0
force_recalculate = False  # Установите True, чтобы пересчитать все заново


# Определение путей к данным (относительные пути)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Создаем папки для результатов, если их нет
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'calculations'), exist_ok=True)

folder_path_A = os.path.join(DATA_DIR, 'A')
folder_path_B = os.path.join(DATA_DIR, 'B')
folder_path_E = os.path.join(DATA_DIR, 'E')

# Проверяем существование папок
for path in [folder_path_A, folder_path_B, folder_path_E]:
    if not os.path.exists(path):
        print(f"Предупреждение: папка {path} не существует")

# Буфер для сохранения результатов
results_buffer = []
results_buffer.append(f"РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
results_buffer.append(f"Параметры алгоритма: ants={num_ants}, iterations={num_iterations}, alpha={alpha}, beta={beta}, evaporation={evaporation_rate}")
results_buffer.append(f"Force recalculate: {force_recalculate}")
results_buffer.append("="*60)


# Обработка наборов данных
print("Обработка набора A")
deviations_A, times_A, dimensions_A = process_folder_with_aco(
    folder_path_A, num_ants, num_iterations, alpha, beta, evaporation_rate, initial_pheromone,
    RESULTS_DIR, results_buffer, force_recalculate
)

print("\nОбработка набора B")
deviations_B, times_B, dimensions_B = process_folder_with_aco(
    folder_path_B, num_ants, num_iterations, alpha, beta, evaporation_rate, initial_pheromone,
    RESULTS_DIR, results_buffer, force_recalculate
)

print("\nОбработка набора E")
deviations_E, times_E, dimensions_E = process_folder_with_aco(
    folder_path_E, num_ants, num_iterations, alpha, beta, evaporation_rate, initial_pheromone,
    RESULTS_DIR, results_buffer, force_recalculate
)


# Вывод статистики по каждому набору
print_set_statistics("A", deviations_A, times_A, dimensions_A, results_buffer)
print_set_statistics("B", deviations_B, times_B, dimensions_B, results_buffer)
print_set_statistics("E", deviations_E, times_E, dimensions_E, results_buffer)


# Сохраняем результаты в файл
results_file = save_results_to_file(RESULTS_DIR, "all_sets", "\n".join(results_buffer))
print(f"\nРезультаты сохранены в файл: {results_file}")


# Построение и сохранение графиков
plt.figure(figsize=(14, 6))

# Агрегируем данные
dims_A_agg, devs_A_agg, times_A_agg = aggregate_by_dimension(dimensions_A, deviations_A, times_A)
dims_B_agg, devs_B_agg, times_B_agg = aggregate_by_dimension(dimensions_B, deviations_B, times_B)
dims_E_agg, devs_E_agg, times_E_agg = aggregate_by_dimension(dimensions_E, deviations_E, times_E)

# График 1: Зависимость качества решения от размерности задачи
plt.subplot(1, 2, 1)
if dims_A_agg:
    plt.plot(dims_A_agg, devs_A_agg, 'o-', label='Set A', linewidth=2, markersize=8)
if dims_B_agg:
    plt.plot(dims_B_agg, devs_B_agg, 's-', label='Set B', linewidth=2, markersize=8)
if dims_E_agg:
    plt.plot(dims_E_agg, devs_E_agg, 'd-', label='Set E', linewidth=2, markersize=8)
plt.xlabel('Размерность задачи', fontsize=12)
plt.ylabel('Отклонение от оптимальных решений (%)', fontsize=12)
plt.title('Зависимость качества решения от размерности задачи', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# График 2: Зависимость времени решения от размерности задачи
plt.subplot(1, 2, 2)
if dims_A_agg:
    plt.plot(dims_A_agg, times_A_agg, 'o-', label='Set A', linewidth=2, markersize=8)
if dims_B_agg:
    plt.plot(dims_B_agg, times_B_agg, 's-', label='Set B', linewidth=2, markersize=8)
if dims_E_agg:
    plt.plot(dims_E_agg, times_E_agg, 'd-', label='Set E', linewidth=2, markersize=8)
plt.xlabel('Размерность задачи', fontsize=12)
plt.ylabel('Время выполнения (секунды)', fontsize=12)
plt.title('Зависимость времени решения от размерности задачи', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

plt.tight_layout()

# Сохраняем графики
plots_dir = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'), dpi=300, bbox_inches='tight')
print(f"Графики сохранены в папку: {plots_dir}")

plt.show()
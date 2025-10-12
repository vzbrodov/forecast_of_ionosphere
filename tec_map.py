#!/usr/bin/env python3
"""
Скрипт для построения карты TEC из коэффициентов сферических гармоник
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import sys
import os

# Добавляем путь к модулям
sys.path.append('/home/fanat/work/forecast_of_ionosphere')

from spherical_harmonics import SphericalHarmonics

def load_latest_data(data_file, n_records=1):
    """
    Загрузка последних n записей из файла данных
    
    Args:
        data_file (str): Путь к файлу с данными
        n_records (int): Количество последних записей для загрузки
        
    Returns:
        tuple: (dates, coefficients)
    """
    print(f"Загрузка последних {n_records} записей из {data_file}...")
    
    # Читаем данные как numpy array
    raw_data = np.loadtxt(data_file)
    
    # Берем последние n_records записей
    if n_records == 1:
        dates = raw_data[-1, 0]
        coefficients = raw_data[-1, 1:]
    else:
        dates = raw_data[-n_records:, 0]
        coefficients = raw_data[-n_records:, 1:]
    
    print(f"Загружено {n_records} записей")
    print(f"Количество коэффициентов: {coefficients.shape[-1]}")
    
    # Преобразуем даты в datetime
    if n_records == 1:
        date_datetime = datetime(1858, 11, 17) + timedelta(days=dates)
        print(f"Дата последних данных: {date_datetime}")
        return date_datetime, coefficients
    else:
        dates_datetime = [datetime(1858, 11, 17) + timedelta(days=d) for d in dates]
        print(f"Период данных: {dates_datetime[0]} - {dates_datetime[-1]}")
        return dates_datetime, coefficients

def create_tec_map(coefficients, title="TEC Distribution", save_path=None, scale_factor=30.0):
    """
    Создание карты TEC из коэффициентов сферических гармоник
    
    Args:
        coefficients (array): Массив коэффициентов
        title (str): Заголовок карты
        save_path (str): Путь для сохранения
        scale_factor (float): Масштабирующий коэффициент для приведения к реальным значениям TEC
        
    Returns:
        tuple: (fig, ax)
    """
    print("Создание карты TEC...")
    
    # Масштабируем коэффициенты для получения реалистичных значений TEC
    scaled_coefficients = coefficients / scale_factor
    print(f"Коэффициенты масштабированы на {scale_factor}")
    
    # Инициализируем класс для работы со сферическими гармониками
    n_max = 15  # Максимальная степень разложения
    spherical_harmonics = SphericalHarmonics(n_max)
    
    # Создаем высокоразрешающую сетку
    lon = np.linspace(-180, 180, 360)  # 1 градус
    lat = np.linspace(-90, 90, 180)    # 1 градус
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Восстанавливаем ПЭС
    print("Восстановление ПЭС из коэффициентов...")
    tect_values = spherical_harmonics.reconstruct_tect(scaled_coefficients, lon_grid, lat_grid)
    
    # Создаем карту
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Добавляем географические элементы
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    ax.add_feature(cfeature.LAND, alpha=0.1, color='lightgray')
    
    # Отображаем данные
    vmin = np.percentile(tect_values, 5)
    vmax = np.percentile(tect_values, 95)
    
    print(f"Диапазон значений TEC: {np.min(tect_values):.2f} - {np.max(tect_values):.2f} TECU")
    print(f"Центральный диапазон (5%-95%): {vmin:.2f} - {vmax:.2f} TECU")
    
    im = ax.contourf(lon_grid, lat_grid, tect_values, 
                    levels=50, cmap='plasma', 
                    transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax)
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('TEC (TECU)', fontsize=14, fontweight='bold')
    
    # Настройка карты
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                alpha=0.5, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Карта сохранена: {save_path}")
    
    return fig, ax

def analyze_coefficients(coefficients):
    """
    Анализ коэффициентов сферических гармоник
    
    Args:
        coefficients (array): Массив коэффициентов
    """
    print("\nАнализ коэффициентов сферических гармоник:")
    print(f"Общее количество коэффициентов: {len(coefficients)}")
    print(f"Среднее значение: {np.mean(coefficients):.2f}")
    print(f"Стандартное отклонение: {np.std(coefficients):.2f}")
    print(f"Минимальное значение: {np.min(coefficients):.2f}")
    print(f"Максимальное значение: {np.max(coefficients):.2f}")
    
    # Анализ по типам гармоник
    spherical_harmonics = SphericalHarmonics(15)
    
    zonal_coeffs = []
    sectoral_coeffs = []
    tesseral_coeffs = []
    
    for i, coeff in enumerate(coefficients):
        n, k, p, harmonic_type = spherical_harmonics.get_coefficient_info(i)
        if harmonic_type == "zonal":
            zonal_coeffs.append(coeff)
        elif harmonic_type == "sectoral":
            sectoral_coeffs.append(coeff)
        elif harmonic_type == "tesseral":
            tesseral_coeffs.append(coeff)
    
    print(f"\nЗональные гармоники (k=0): {len(zonal_coeffs)} коэффициентов")
    if zonal_coeffs:
        print(f"  Среднее: {np.mean(zonal_coeffs):.2f}, Стд: {np.std(zonal_coeffs):.2f}")
    
    print(f"Секторальные гармоники (k=n): {len(sectoral_coeffs)} коэффициентов")
    if sectoral_coeffs:
        print(f"  Среднее: {np.mean(sectoral_coeffs):.2f}, Стд: {np.std(sectoral_coeffs):.2f}")
    
    print(f"Тессеральные гармоники (0<k<n): {len(tesseral_coeffs)} коэффициентов")
    if tesseral_coeffs:
        print(f"  Среднее: {np.mean(tesseral_coeffs):.2f}, Стд: {np.std(tesseral_coeffs):.2f}")

def main():
    """Основная функция"""
    print("=== Построение карты TEC из коэффициентов сферических гармоник ===\n")
    
    # Путь к файлу данных
    data_file = '/home/fanat/work/forecast_of_ionosphere/1999-2022rawdata'
    
    # Загружаем последние данные
    date, coefficients = load_latest_data(data_file, n_records=1)
    
    # Анализируем коэффициенты
    analyze_coefficients(coefficients)
    
    # Создаем карту TEC
    title = f"TEC Distribution - {date.strftime('%Y-%m-%d %H:%M')}"
    save_path = '/home/fanat/work/forecast_of_ionosphere/tec_map_latest.png'
    
    fig, ax = create_tec_map(coefficients, title, save_path)
    
    # Проверяем асимметрию
    print("\nПроверка асимметрии карты:")
    from spherical_harmonics import SphericalHarmonics
    import numpy as np
    
    # Создаем тестовую сетку для проверки
    lon_test = np.linspace(-180, 180, 72)
    lat_test = np.linspace(-90, 90, 36)
    lon_grid_test, lat_grid_test = np.meshgrid(lon_test, lat_test)
    
    sh = SphericalHarmonics(15)
    # Используем тот же масштабирующий коэффициент
    scaled_coefficients_test = coefficients / 30.0
    tect_values_test = sh.reconstruct_tect(scaled_coefficients_test, lon_grid_test, lat_grid_test)
    
    # Проверяем асимметрию
    lat_30_idx = np.argmin(np.abs(lat_test - 30))
    lat_minus30_idx = np.argmin(np.abs(lat_test - (-30)))
    
    values_30 = tect_values_test[lat_30_idx, :]
    values_minus30 = tect_values_test[lat_minus30_idx, :]
    
    print(f"Широта +30°: среднее={np.mean(values_30):.2f} TECU")
    print(f"Широта -30°: среднее={np.mean(values_minus30):.2f} TECU")
    print(f"Разность: {np.mean(values_30) - np.mean(values_minus30):.2f} TECU")
    
    if abs(np.mean(values_30) - np.mean(values_minus30)) > 50:
        print("✓ Карта асимметрична относительно экватора")
    else:
        print("⚠ Карта может быть симметрична относительно экватора")
    
    # Показываем карту
    plt.show()
    
    print(f"\n✓ Карта TEC успешно создана и сохранена: {save_path}")
    
    return fig, ax, coefficients

if __name__ == "__main__":
    fig, ax, coefficients = main()
 
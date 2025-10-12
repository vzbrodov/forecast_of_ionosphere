#!/usr/bin/env python3
"""
Модуль для работы со сферическими гармониками и восстановления ПЭС
"""

import numpy as np
from scipy.special import lpmn, factorial
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class SphericalHarmonics:
    """Класс для работы со сферическими гармониками"""
    
    def __init__(self, n_max=15):
        """
        Инициализация
        
        Args:
            n_max (int): Максимальная степень разложения
        """
        self.n_max = n_max
        self.normalization_factors = self._compute_normalization_factors()
    
    def _compute_normalization_factors(self):
        """Вычисление нормирующих коэффициентов"""
        factors = {}
        for n in range(self.n_max + 1):
            for k in range(n + 1):
                # Нормирующий коэффициент для присоединенных полиномов Лежандра
                if k == 0:
                    factors[(n, k)] = np.sqrt(2 * n + 1)
                else:
                    factors[(n, k)] = np.sqrt(2 * (2 * n + 1) * factorial(n - k) / factorial(n + k))
        return factors
    
    def get_coefficient_info(self, i):
        """
        Получение информации о коэффициенте по его индексу
        
        Args:
            i (int): Индекс коэффициента
            
        Returns:
            tuple: (n, k, p, type_name)
        """
        for n in range(self.n_max + 1):
            for k in range(n + 1):
                for p in [0, 1]:
                    if i == n**2 + 2*k + p - 1:
                        if k == 0:
                            harmonic_type = "zonal"
                        elif k == n:
                            harmonic_type = "sectoral"
                        else:
                            harmonic_type = "tesseral"
                        return n, k, p, harmonic_type
        return None, None, None, None
    
    def compute_legendre_polynomials(self, latitude_rad):
        """
        Вычисление нормированных полиномов Лежандра от sin φ
        
        Args:
            latitude_rad (array): Широты в радианах
            
        Returns:
            dict: Словарь с полиномами Лежандра
        """
        # Используем scipy для вычисления присоединенных полиномов Лежандра
        sin_lat = np.sin(latitude_rad)  # Используем sin φ как аргумент
        
        legendre_polys = {}
        
        for n in range(self.n_max + 1):
            for k in range(n + 1):
                # Вычисляем присоединенные полиномы Лежандра от sin φ
                # lpmn возвращает массив размером (m+1, n+1), где m - максимальный порядок, n - максимальная степень
                Pmn_array = lpmn(n, n, sin_lat)[0]  # Вычисляем все полиномы до степени n и порядка n
                Pmn = Pmn_array[k, n]  # Берем полином степени n и порядка k
                
                # Применяем нормировку
                norm_factor = self.normalization_factors.get((n, k), 1.0)
                legendre_polys[(n, k)] = Pmn * norm_factor
        
        return legendre_polys
    
    def reconstruct_tect(self, coefficients, longitude_grid, latitude_grid):
        """
        Восстановление ПЭС по коэффициентам сферических гармоник
        
        Args:
            coefficients (array): Массив коэффициентов
            longitude_grid (array): Сетка долгот в градусах
            latitude_grid (array): Сетка широт в градусах
            
        Returns:
            array: Восстановленные значения ПЭС
        """
        # Преобразуем в радианы
        lon_rad = longitude_grid * np.pi / 180
        lat_rad = latitude_grid * np.pi / 180
        
        # Вычисляем полиномы Лежандра
        legendre_polys = self.compute_legendre_polynomials(lat_rad)
        
        # Инициализируем результат
        tect_values = np.zeros_like(longitude_grid)
        
        # Восстанавливаем ПЭС согласно вашей формуле:
        # y(λ, φ) = Σ Σ P̃_nk sin φ (a_nk cos kλ + b_nk sin kλ)
        # где P̃_nk sin φ означает полином Лежандра от sin φ
        # p=0 соответствует sin kλ, p=1 соответствует cos kλ
        
        for i, coeff in enumerate(coefficients):
            n, k, p, _ = self.get_coefficient_info(i)
            if n is None or coeff == 0:
                continue
            
            # Получаем полином Лежандра (вычислен от sin φ)
            P_nk = legendre_polys.get((n, k), 0)
            
            if k == 0:  # Зональные гармоники
                # Для зональных гармоник: coeff * P̃_nk(sin φ)
                tect_values += coeff * P_nk
            else:  # Тессеральные и секторальные гармоники
                if p == 0:  # sin kλ
                    tect_values += coeff * P_nk * np.sin(k * lon_rad)
                else:  # cos kλ
                    tect_values += coeff * P_nk * np.cos(k * lon_rad)
        
        return tect_values
    
    def create_global_heatmap(self, coefficients, title="TEC Distribution", 
                            save_path=None, vmin=None, vmax=None):
        """
        Создание глобальной тепловой карты распределения ПЭС
        
        Args:
            coefficients (array): Массив коэффициентов
            title (str): Заголовок карты
            save_path (str): Путь для сохранения
            vmin, vmax (float): Границы цветовой шкалы
            
        Returns:
            tuple: (fig, ax)
        """
        # Создаем высокоразрешающую сетку
        lon = np.linspace(-180, 180, 720)  # 0.5 градуса
        lat = np.linspace(-90, 90, 360)    # 0.5 градуса
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Восстанавливаем ПЭС
        print("Восстановление ПЭС...")
        tect_values = self.reconstruct_tect(coefficients, lon_grid, lat_grid)
        
        # Создаем карту
        fig = plt.figure(figsize=(20, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Добавляем географические элементы
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
        ax.add_feature(cfeature.LAND, alpha=0.1, color='lightgray')
        
        # Отображаем данные
        if vmin is None:
            vmin = np.percentile(tect_values, 5)
        if vmax is None:
            vmax = np.percentile(tect_values, 95)
        
        # Используем кастомную цветовую схему: синий -> желтый -> красный
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000']
        n_bins = 100
        cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        im = ax.contourf(lon_grid, lat_grid, tect_values, 
                        levels=50, cmap=cmap_custom, 
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
    
    def analyze_harmonic_structure(self, coefficients):
        """
        Анализ структуры гармоник
        
        Args:
            coefficients (array): Массив коэффициентов
            
        Returns:
            dict: Результаты анализа
        """
        harmonic_analysis = {
            'zonal': {'indices': [], 'coefficients': [], 'powers': []},
            'sectoral': {'indices': [], 'coefficients': [], 'powers': []},
            'tesseral': {'indices': [], 'coefficients': [], 'powers': []}
        }
        
        for i, coeff in enumerate(coefficients):
            n, k, p, harmonic_type = self.get_coefficient_info(i)
            if n is None:
                continue
            
            harmonic_analysis[harmonic_type]['indices'].append(i)
            harmonic_analysis[harmonic_type]['coefficients'].append(coeff)
            harmonic_analysis[harmonic_type]['powers'].append(coeff**2)
        
        return harmonic_analysis
    
    def plot_harmonic_spectrum(self, coefficients, save_path=None):
        """
        Построение спектра гармоник
        
        Args:
            coefficients (array): Массив коэффициентов
            save_path (str): Путь для сохранения
        """
        analysis = self.analyze_harmonic_structure(coefficients)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Спектр сферических гармоник', fontsize=16, fontweight='bold')
        
        # Зональные гармоники
        axes[0, 0].bar(analysis['zonal']['indices'], 
                      analysis['zonal']['powers'], 
                      color='blue', alpha=0.7)
        axes[0, 0].set_title('Зональные гармоники (k=0)')
        axes[0, 0].set_xlabel('Индекс коэффициента')
        axes[0, 0].set_ylabel('Мощность')
        axes[0, 0].set_yscale('log')
        
        # Секторальные гармоники
        axes[0, 1].bar(analysis['sectoral']['indices'], 
                      analysis['sectoral']['powers'], 
                      color='red', alpha=0.7)
        axes[0, 1].set_title('Секторальные гармоники (k=n)')
        axes[0, 1].set_xlabel('Индекс коэффициента')
        axes[0, 1].set_ylabel('Мощность')
        axes[0, 1].set_yscale('log')
        
        # Тессеральные гармоники
        axes[1, 0].bar(analysis['tesseral']['indices'], 
                      analysis['tesseral']['powers'], 
                      color='green', alpha=0.7)
        axes[1, 0].set_title('Тессеральные гармоники (0<k<n)')
        axes[1, 0].set_xlabel('Индекс коэффициента')
        axes[1, 0].set_ylabel('Мощность')
        axes[1, 0].set_yscale('log')
        
        # Общий спектр
        all_indices = list(range(len(coefficients)))
        all_powers = [c**2 for c in coefficients]
        axes[1, 1].bar(all_indices, all_powers, alpha=0.5)
        axes[1, 1].set_title('Общий спектр')
        axes[1, 1].set_xlabel('Индекс коэффициента')
        axes[1, 1].set_ylabel('Мощность')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Спектр сохранен: {save_path}")
        
        return fig, axes

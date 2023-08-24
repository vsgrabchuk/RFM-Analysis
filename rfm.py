import pandas as pd
from pprint import pprint


def get_rfm_values(
    df,
    unit_id,
    money,
    time,
    now,
    time_horizon
):
    '''
    Функция считает метрики recency, frequency и monetary для каждого unit_id в df
    
    Parameters
    ----------
    df: pandas.DataFrame
        Исходный df без пропущенных значений
    unit_id: str
        Название столбца с id для каждого из которых необходимо считать значения recency, frequency и monetary
        Например, customer_id
    money: str
        Название столбца с суммой по каждой покупке unit_id
    time: str
        Название столбца с датой по каждой покупке unit_id
    now: pandas.datetime
        Текущие дата и время
    time_horizon: int
        Временной горизонт, на котором считаются метрики (количество рассматриваемых дней от текущего)
    
    Returns
    -------
    rfm_df: pandas.DataFrame
        df с добавленными столбцами (recency, frequency и monetary)
    '''
    # Фильтрация данных по временному горизонту
    start_time = now - pd.Timedelta(days=time_horizon)
    df = df.query('@start_time <= ' + time + ' and ' + time + ' <= @now')
    # Добавление колонки с количеством дней до текущего момента
    df['days_before'] = (now - df[time]).dt.days
    # Расчёт метрик
    agg_exp = {  # Агрегирующие выражения
        'days_before': 'min',     # Количество дней с последней покупки пользователем (recency)
        time: 'count',            # Количество заказов пользователя (frequency)
        money: 'sum'              # Доход от пользователя (monetary)
    }
    new_columns = {  # Новые названия столбцов
        'days_before':       'recency',
        'order_approved_at': 'frequency',
        money:               'monetary'
    }
    rfm_df = (
        df.groupby(unit_id, as_index=False).agg(agg_exp)
        .rename(columns=new_columns)
    )
    
    return rfm_df


def get_rfm_score(
    df,
    score_column,
    score_name,
    max_score=3,
    add_score_bins=False,
    auto_max_score_adjust=False,
    print_info=True,
    round_info_val=3
):
    '''
    Функция для оценивания параметров recency, frequency и monetary в дискретном диапазоне [1, max_score]
    с помощью автоподбора квантилей (т.е. в автоматическом режиме оценки присваиваются сегментам примерно равного размера)

    Arguments
    ---------
    df: pd.DataFrame
        Исходный df с рассчитанной метрикой recency || frequency || monetary для каждого unit_id
    score_column: str
        Название столбца для оценивания
    score_name: str
        Название требующейся оценки
        'r' => recency
        'f' => frequency
        'm' => monetary
    max_score: int, default 3
        Максимальная оценка параметра на дискретном интервале [1, max_score]
    auto_max_score_adjust: bool, default False
        Автоматическая корректировка max_score (уменьшение) при совпадении значений квантилей
    add_score_bins: bool, default False
        Добавить к возвращаемому df столбец с информацией по диапазону значений из score_column, 
        соответствующих оценке
        Пояснение по интервалам: 
            [a_1, b_1] => score_1
            (b_1, b_2] => score_2
            ...
            (b_{n-1}, b_n] => score_{n-1}
            (b_n b_{n+1}] => score_n
    print_info: bool, default True
        Флаг для вывода информации
        Пояснение по интервалам:
            [a_1, b_1] => score_1
            (b_1, b_2] => score_2
            ...
            (b_{n-1}, b_n] => score_{n-1}
            (b_n b_{n+1}] => score_n
    round_info_val: int, default 3
        Число знаков после запятой для отображаемой информации

    Returns
    -------
    score_df: pd.DataFrame
        df со столбцом, где записаны оценки для параметра
    '''
    # Полное название используемой метрики
    if score_name == 'r':
        metric = 'recency'
    elif score_name == 'f':
        metric = 'frequency'
    elif score_name == 'm':
        metric = 'monetary'
    else:
        metric = None
    
    # Столбец со значениями для оценивания
    vals = df[score_column]
    
    if max_score > 1:
        # Квантили и соответствующие им значения
        d_quantile = 1 / max_score  # Межквантильный интервал
        quintile_list = [d_quantile*i for i in range(1, max_score)]  # Список квантилей
        quantiles = vals.quantile(quintile_list).tolist()  # Список значений квантилей
        
        # Случай дублирования квантилей
        if len(quantiles) != len(set(quantiles)):
            if auto_max_score_adjust:
                print(f'AUTO ADJUSTMENT max_score={max_score} ->', max_score-1)
                return get_rfm_score(
                    df=df,
                    score_column=score_column,
                    score_name=score_name,
                    max_score=max_score-1,
                    add_score_bins=add_score_bins,
                    auto_max_score_adjust=auto_max_score_adjust,
                    print_info=print_info,
                    round_info_val=round_info_val
                )
            else:
                quintile_dict = ( # Словарь с квантилями (для вывода информации)
                    vals
                    .quantile(quintile_list)
                    .to_frame().reset_index().round(round_info_val)
                    .rename(columns={vals.name: metric + '_quantiles'})
                    .set_index('index')
                    .to_dict()
                )
                raise QuantileDuplicate(quintile_dict)  # Отображение исключения

        scores_list = [i for i in range(1, max_score+1)]  # Список оценок
        bins_cutoffs = [vals.min()] + quantiles + [vals.max()]  # Список отсечек для разбиения
                                                                # данных на промежутки и
                                                                # получения оценок
        # Присвоение оценок исходным значениям
        if score_name == 'r':
            scores_list.reverse()  # Оценки в обратном порядке (чем меньше абсолютнное значение recency, тем лучше, с оценкой - наоборот)
            scores = pd.cut(
                x=vals, 
                bins=[bins_cutoffs[0]-1] + bins_cutoffs[1:-1] + [bins_cutoffs[-1]+1], 
                labels=scores_list, 
                include_lowest=True).astype(str)
        else:
            scores = pd.cut(
                x=vals, 
                bins=[bins_cutoffs[0]-1] + bins_cutoffs[1:-1] + [bins_cutoffs[-1]+1], 
                labels=scores_list, 
                include_lowest=True).astype(str)
    else:  # max_score == 1
        scores_list = ['1']
        bins_cutoffs =  [vals.min(), vals.max()]
        scores = pd.Series(['1' for _ in range(len(vals))])
    
    # Добавление столбца с оценками
    df[score_name] = scores
    
    
    # Информация по диапазону значений из score_column, соответствующих оценке
    bins_for_scores = {}
    for i, score in enumerate(scores_list):
            bins_for_scores[str(score)] = str([round(bins_cutoffs[i], round_info_val), 
                                               round(bins_cutoffs[i+1], round_info_val)])

    # Добавление к возвращаемому df столбца с информацией по диапазону значений из score_column, 
    # соответствующих оценке
    if add_score_bins:
        df[score_name + '_bin'] = scores.replace(bins_for_scores)
    
    # Вывод информации
    if print_info:
        # Информация по диапазону параметров для оценок
        pprint({'bins_for_scores': bins_for_scores})
    
    return df


class QuantileDuplicate(Exception):
    '''Класс для отображения исключения в случае дублирования значений квантилей'''
    pass


def get_rfm_agg(
    df,
    unit_id,
    r_name='r',
    f_name='f',
    m_name='m',
    use_bins=False,
    r_bin_name='r_bin',
    f_bin_name='f_bin',
    m_bin_name='m_bin'
):
    '''
    Функция для получения агрегированной таблицы с подсчётом количества unit_id в каждом rfm сегменте
    
    Parameters
    ----------
    df: pandas.DataFrame
        Исходный df с оцененными метриками r, f, m
    unit_id: str
        Название столбца для подсчёта количества градаций его значений
    r_name: str, default 'r'
        Название столбца с оценкой recency: r
    f_name: str, default 'f'
        Название столбца с оценкой frequency: f
    m_name: str, default 'm'
        Название столбца с оценкой monetary: m
    use_bins: bool, default False,
        Флаг отображения в итоговой таблице диапазона значений метрик для оценок
    r_bins_name: str, default 'r_bins'
        Название столбца с диапазоном метрик оценки recency
    f_bins_name: str, default 'f_bins'
        Название столбца с диапазоном метрик оценки frequency
    m_bins_name: str, default 'm_bins'
        Название столбца с диапазоном метрик оценки monetary
    
    Returns
    -------
    rfm_df: pandas.DataFrame
        Итоговый df с rfm сегментами и количеством unit_id в каждом из них
    '''
    df['rfm'] = df[r_name] + df[f_name] + df[m_name]
    
    if use_bins:
        rfm_df = (
            df.
            groupby(['rfm', r_bin_name, f_bin_name, m_bin_name], as_index=False)[unit_id].count()
            .rename(columns={unit_id: 'amount'})
        )
    else:
        rfm_df = (
            df.
            groupby('rfm', as_index=False)[unit_id].count()
            .rename(columns={unit_id: 'amount'})
        )
        
    
    return rfm_df.sort_values('rfm', ascending=False).reset_index(drop=True)
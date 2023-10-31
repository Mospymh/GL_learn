import datetime
import time
from typing import Optional, Tuple
from datetime import date, timedelta
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


class TimeDateUtil(object):
    @staticmethod
    def convert_str_to_date(src_str: str, src_format: str) -> Optional[datetime.datetime]:
        try:
            d = datetime.datetime.strptime(src_str, src_format)
        except ValueError:
            d = None
        except TypeError:
            d = None
        return d

    @staticmethod
    def convert_date_to_str(src_date: datetime.datetime, tgt_format: str) -> str:
        return src_date.strftime(tgt_format)

    @staticmethod
    def convert_format(src_str: str, src_format: str, tgt_format: str) -> Optional[str]:
        src_date = TimeDateUtil.convert_str_to_date(src_str, src_format)
        if src_date is None:
            return None
        return TimeDateUtil.convert_date_to_str(src_date, tgt_format)

    @staticmethod
    def is_valid_date_format(src_str: str, src_format: str) -> bool:
        return TimeDateUtil.convert_str_to_date(src_str, src_format) is not None

    @staticmethod
    def get_cur_date_str(tgt_format: str) -> Optional[str]:
        now = datetime.datetime.now()
        return TimeDateUtil.convert_date_to_str(now, tgt_format)

    @staticmethod
    def get_last_days_date(tgt_format: str, n: int) -> str:
        last_days = date.today() - timedelta(n)
        return last_days.strftime(tgt_format)

    @staticmethod
    def get_previous_date_str(src_str: str, src_format: str, tgt_format: str, n: int) -> str:
        last_days = TimeDateUtil.convert_str_to_date(src_str, src_format) - timedelta(n)
        return last_days.strftime(tgt_format)

    @staticmethod
    def get_current_timestamp() -> float:
        return time.time()

    @staticmethod
    def get_timestamp_arr(arr: np.ndarray, interpolation: bool = False) -> np.ndarray:
        """
        :param arr: '2016-01-01'
        :param interpolation:
        :return: 736332.0
        """
        if len(arr) == 0:
            return np.empty([0, 0])
        if not interpolation:
            return np.vectorize(lambda dt: mdates.datestr2num(dt))(arr)
        else:
            first_val = mdates.datestr2num(arr[0])
            return np.array([first_val + idx for idx, i in enumerate(arr)])

    @staticmethod
    def get_time_delta(src_str: str, tgt_str: str, src_format: str, tgt_format: str, unit: str = 'DAY') -> int:
        last_days = TimeDateUtil.convert_str_to_date(src_str, src_format) - \
                    TimeDateUtil.convert_str_to_date(tgt_str, tgt_format)
        if unit == 'DAY':
            return last_days.days
        elif unit == 'SEC':
            return last_days.seconds
        return last_days.days

    @staticmethod
    def extract_year_month_day(date_str: str, date_format: str) -> Tuple[int ,int , int]:
        dt = TimeDateUtil.convert_str_to_date(date_str, date_format)
        if dt is None:
            return None, None, None
        return dt.year, dt.month, dt.day

    @staticmethod
    def extract_weekday(date_str: str, date_format: str) -> int:
        dt = TimeDateUtil.convert_str_to_date(date_str, date_format)
        return dt.weekday()

    @staticmethod
    def compute_fiscal_date(year: int, month: int) -> str:
        publish_date = "{0}-09-30".format(year) if month == 9 else \
                       "{0}-06-30".format(year) if month == 6 else \
                       "{0}-03-31".format(year) if month == 3 else \
                       "{0}-12-31".format(year)
        return publish_date

    @staticmethod
    def comp_ma(
            fiscal_df: pd.DataFrame,
            factor_name: str,
            date_format: str,
            window: int = 4
    ) -> pd.DataFrame:
        """
        注意nan值的处理
        针对资产负债表计算MEAN
        :return:
        """
        def compute_mean(
                idxs: np.ndarray,
                df: pd.DataFrame,
                factor_name: str,
                window: int
        ) -> float:
            part_df = df.iloc[list(map(int, idxs))]
            if len(part_df.index) == 0:
                return np.nan
            last_publish_date = part_df['PUBLISH_DATE'].iloc[-1]
            last_end_date = part_df['END_DATE'].iloc[-1]
            part_df = part_df[(part_df['END_DATE'] <= last_end_date) & (part_df['PUBLISH_DATE'] <= last_publish_date)] \
                .drop_duplicates(subset=['TICKER_SYMBOL', 'END_DATE'], keep='last') \
                .sort_values(by=['TICKER_SYMBOL', 'END_DATE', 'PUBLISH_DATE'], ascending=[True, True, True]) \
                .dropna(subset=[factor_name], axis=0)
            return np.mean(part_df.drop_duplicates(subset=['YEAR', 'MONTH'], keep='last').tail(window)[factor_name])

        factor_name_mean = "{0}_{1}".format(factor_name, "MA")
        fiscal_df['YEAR'] = fiscal_df['END_DATE'].apply(lambda x: TimeDateUtil.convert_str_to_date(x, date_format).year)
        fiscal_df['MONTH'] = fiscal_df['END_DATE'].apply(lambda x: TimeDateUtil.convert_str_to_date(x, date_format).month)
        # 均值
        fiscal_df = fiscal_df.sort_values(by=['TICKER_SYMBOL', 'PUBLISH_DATE', 'END_DATE'], ascending=[True, True, True])
        tot = len(fiscal_df.index)
        fiscal_df['IDX'] = range(tot)
        fiscal_df[factor_name_mean] = fiscal_df['IDX'].rolling(window=tot, min_periods=0, center=False) \
            .apply(lambda x: compute_mean(x, fiscal_df, factor_name, window), raw=True)
        # 防止某一天仅修正上期的财报
        fiscal_df = fiscal_df.sort_values(by=['TICKER_SYMBOL', 'END_DATE', 'PUBLISH_DATE'], ascending=[True, True, True])
        return fiscal_df.drop(['YEAR', 'MONTH', 'IDX'], axis=1)

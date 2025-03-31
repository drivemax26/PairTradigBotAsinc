import asyncio
import aiohttp
import json
import math
import logging
import ssl
from datetime import datetime
import pandas as pd
from binance import AsyncClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_interval(interval_str):
    """
    Преобразует строковый таймфрейм (например, "1m", "1h", "1d") в количество секунд.
    """
    unit = interval_str[-1]
    number = int(interval_str[:-1])
    if unit == 'm':
        return number * 60
    elif unit == 'h':
        return number * 3600
    elif unit == 'd':
        return number * 86400
    else:
        raise ValueError("Unsupported interval format")


async def send_telegram_message(message, telegram_bot_token, telegram_chat_id):
    """
    Асинхронная отправка сообщения в Telegram.
    """
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Ошибка отправки Telegram сообщения: {await resp.text()}")
                else:
                    logger.info("Сообщение отправлено в Telegram")
    except Exception as e:
        logger.error(f"Ошибка отправки Telegram сообщения: {e}")


async def get_symbol_step_size_async(client, symbol):
    """
    Получает stepSize для указанного символа через асинхронного клиента Binance.
    """
    info = await client.futures_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            for f in s["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    return float(f["stepSize"])
    return None


def round_quantity(quantity, step_size):
    """
    Округление количества согласно шагу (stepSize).
    """
    decimals = int(round(-math.log(step_size, 10), 0))
    return round(quantity, decimals)


def calculate_indicators(closed_prices, ma_period, sigma1, sigma2, sigma3):
    """
    Рассчитывает MA и уровни Bollinger Bands по последним закрытым свечам.
    """
    if len(closed_prices) < ma_period:
        return None
    series = pd.Series(closed_prices[-ma_period:])
    ma = series.mean()
    std = series.std()
    indicators = {
        "MA": ma,
        "std": std,
        "upper_band_1": ma + sigma1 * std,
        "lower_band_1": ma - sigma1 * std,
        "upper_band_2": ma + sigma2 * std,
        "lower_band_2": ma - sigma2 * std,
        "upper_band_3": ma + sigma3 * std,
        "lower_band_3": ma - sigma3 * std,
    }
    return indicators


class AsyncTradingBot:
    def __init__(self, config, client):
        self.config = config
        self.client = client
        self.symbol1 = config["symbol1"]
        self.symbol2 = config["symbol2"]
        self.candle_interval = config["candle_interval"]
        self.candle_interval_seconds = parse_interval(self.candle_interval)
        self.ma_period = config["ma_period"]
        self.sigma1 = config["first_sigma"]
        self.sigma2 = config["second_sigma"]
        self.sigma3 = config["third_sigma"]
        self.order_amount = config["order_amount"]
        self.telegram_bot_token = config["telegram_bot_token"]
        self.telegram_chat_id = config["telegram_chat_id"]

        # Последние полученные цены
        self.last_price_symbol1 = None
        self.last_price_symbol2 = None

        # Текущая свеча для синтетической цены (привязка к UTC)
        self.current_candle = None  # {'start', 'open', 'high', 'low', 'close'}
        self.closed_candles = []  # список закрытых цен синтетических свечей
        self.indicators = None
        self.positions = []  # список открытых позиций

        # Блокировка для защиты проверки и открытия позиций
        self.signal_lock = asyncio.Lock()

        # Флаги, указывающие, что позиция была закрыта по стопу (на 3 сигме)
        self.upper_stop_closed_flag = False
        self.lower_stop_closed_flag = False

    def position_exists(self, direction, sigma_trigger):
        """
        Проверяет, существует ли уже открытая позиция для данного направления и уровня сигмы.
        """
        for pos in self.positions:
            if pos["type"] == direction and pos["sigma_trigger"] == sigma_trigger:
                return True
        return False

    async def load_historical_data(self):
        url = "https://fapi.binance.com/fapi/v1/klines"
        params1 = {"symbol": self.symbol1, "interval": self.candle_interval, "limit": self.ma_period}
        params2 = {"symbol": self.symbol2, "interval": self.candle_interval, "limit": self.ma_period}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params1) as resp1:
                    data1 = await resp1.json()
                async with session.get(url, params=params2) as resp2:
                    data2 = await resp2.json()
            synthetic_prices = []
            length = min(len(data1), len(data2))
            for i in range(length):
                close1 = float(data1[i][4])
                close2 = float(data2[i][4])
                if close2 != 0:
                    synthetic_prices.append(close1 / close2)
            self.closed_candles = synthetic_prices
            logger.info(f"Загружено исторических свечей: {len(self.closed_candles)}")
            self.indicators = calculate_indicators(self.closed_candles, self.ma_period, self.sigma1, self.sigma2, self.sigma3)
            if self.indicators:
                logger.info(
                    f"Индикаторы (исторические данные): MA={self.indicators['MA']:.5f}, "
                    f"+1 сигма={self.indicators['upper_band_1']:.5f}, -1 сигма={self.indicators['lower_band_1']:.5f}, "
                    f"+2 сигма={self.indicators['upper_band_2']:.5f}, -2 сигма={self.indicators['lower_band_2']:.5f}, "
                    f"+3 сигма={self.indicators['upper_band_3']:.5f}, -3 сигма={self.indicators['lower_band_3']:.5f}"
                )
            else:
                logger.info("Не удалось вычислить индикаторы из исторических данных")
        except Exception as e:
            logger.error(f"Ошибка загрузки исторических данных: {e}")

    async def update_price(self, symbol, price):
        now = datetime.utcnow()
        if symbol == self.symbol1:
            self.last_price_symbol1 = price
        elif symbol == self.symbol2:
            self.last_price_symbol2 = price

        if (self.last_price_symbol1 is not None and
            self.last_price_symbol2 is not None and
            self.last_price_symbol2 != 0):
            synthetic_price = self.last_price_symbol1 / self.last_price_symbol2
            logger.info(f"Синтетическая цена: {synthetic_price:.5f}")

            # Обновление текущей свечи
            candle_start = datetime.utcfromtimestamp(
                (now.timestamp() // self.candle_interval_seconds) * self.candle_interval_seconds
            )
            if self.current_candle is None:
                self.current_candle = {'start': candle_start, 'open': synthetic_price, 'high': synthetic_price,
                                        'low': synthetic_price, 'close': synthetic_price}
            elif self.current_candle['start'] != candle_start:
                closed_price = self.current_candle['close']
                self.closed_candles.append(closed_price)
                logger.info(f"Свеча закрыта. Начало: {self.current_candle['start'].strftime('%H:%M:%S')}, Закрытие: {closed_price:.5f}")
                self.indicators = calculate_indicators(self.closed_candles, self.ma_period, self.sigma1, self.sigma2, self.sigma3)
                if self.indicators:
                    logger.info(
                        f"Индикаторы: MA={self.indicators['MA']:.5f}, "
                        f"+1 сигма={self.indicators['upper_band_1']:.5f}, -1 сигма={self.indicators['lower_band_1']:.5f}, "
                        f"+2 сигма={self.indicators['upper_band_2']:.5f}, -2 сигма={self.indicators['lower_band_2']:.5f}, "
                        f"+3 сигма={self.indicators['upper_band_3']:.5f}, -3 сигма={self.indicators['lower_band_3']:.5f}"
                    )
                else:
                    logger.info("Недостаточно данных для расчёта индикаторов.")
                self.current_candle = {'start': candle_start, 'open': synthetic_price, 'high': synthetic_price,
                                        'low': synthetic_price, 'close': synthetic_price}
            else:
                self.current_candle['high'] = max(self.current_candle['high'], synthetic_price)
                self.current_candle['low'] = min(self.current_candle['low'], synthetic_price)
                self.current_candle['close'] = synthetic_price

            # Проверка сигналов в реальном времени
            await self.check_live_signals(synthetic_price)

    async def send_market_order(self, symbol, side, quantity=None):
        """
        Отправка рыночного ордера. Если параметр quantity передан, используется он,
        иначе вычисляется количество по формуле: order_amount / цена.
        """
        try:
            if quantity is None:
                price = self.last_price_symbol1 if symbol == self.symbol1 else self.last_price_symbol2
                if price is None or price <= 0:
                    logger.warning(f"Нет валидной цены для {symbol} при отправке ордера.")
                    return
                step_size = await get_symbol_step_size_async(self.client, symbol)
                if step_size is None:
                    logger.warning(f"Не удалось получить stepSize для {symbol}.")
                    return
                quantity = self.order_amount / price
                quantity = round_quantity(quantity, step_size)
            logger.info(f"Отправка ордера: {side} {symbol}, количество: {quantity:.5f} (используемая цена: {self.last_price_symbol1 if symbol == self.symbol1 else self.last_price_symbol2:.5f})")
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            logger.info(f"Ордер выполнен для {symbol}: {order}")
        except Exception as e:
            logger.error(f"Ошибка при отправке ордера для {symbol}: {e}")

    async def open_position(self, direction, price, sigma_trigger):
        """
        Открытие позиции с сохранением точного количества для каждого символа.
        """
        if self.position_exists(direction, sigma_trigger):
            logger.info(f"Позиция для {direction} с sigma_trigger={sigma_trigger} уже открыта.")
            return

        # Вычисляем количество для каждой сделки по текущим ценам
        if self.last_price_symbol1 is None or self.last_price_symbol2 is None:
            logger.warning("Нет доступных цен для открытия позиции.")
            return

        step_size1 = await get_symbol_step_size_async(self.client, self.symbol1)
        quantity1 = round_quantity(self.order_amount / self.last_price_symbol1, step_size1)

        step_size2 = await get_symbol_step_size_async(self.client, self.symbol2)
        quantity2 = round_quantity(self.order_amount / self.last_price_symbol2, step_size2)

        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        if direction == "upper":
            msg = (f"[Сигнал] {timestamp}: Цена достигла верхней {sigma_trigger} сигмы ({price:.5f}).\n"
                   f"Открытие короткой позиции: ПРОДАЁМ {self.symbol1} на ${self.order_amount} и ПОКУПАЕМ {self.symbol2} на ${self.order_amount}.")
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            await self.send_market_order(self.symbol1, "SELL", quantity1)
            await self.send_market_order(self.symbol2, "BUY", quantity2)
        elif direction == "lower":
            msg = (f"[Сигнал] {timestamp}: Цена достигла нижней {sigma_trigger} сигмы ({price:.5f}).\n"
                   f"Открытие длинной позиции: ПОКУПАЕМ {self.symbol1} на ${self.order_amount} и ПРОДАЁМ {self.symbol2} на ${self.order_amount}.")
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            await self.send_market_order(self.symbol1, "BUY", quantity1)
            await self.send_market_order(self.symbol2, "SELL", quantity2)
        # Сохраняем позицию с точными количествами для каждой сделки
        self.positions.append({
            "type": direction,
            "entry": price,
            "time": datetime.utcnow(),
            "sigma_trigger": sigma_trigger,
            "symbol1_quantity": quantity1,
            "symbol2_quantity": quantity2
        })

    async def close_position(self, pos, price):
        """
        Закрытие позиции с использованием сохранённого точного количества.
        """
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        closed_by_stop = False
        if pos["type"] == "upper":
            profit = pos["entry"] - price
            msg = (f"[Тейкпрофит/Стоплосс] {timestamp}: Цена = {price:.5f}.\n")
            if price >= self.indicators["upper_band_3"]:
                msg += f"Закрытие короткой позиции по стопу. Прибыль: {profit:.5f}"
                closed_by_stop = True
            else:
                msg += f"Закрытие короткой позиции по тейкпрофиту. Прибыль: {profit:.5f}"
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            # При закрытии используем сохранённые количества
            await self.send_market_order(self.symbol1, "BUY", pos["symbol1_quantity"])
            await self.send_market_order(self.symbol2, "SELL", pos["symbol2_quantity"])
            if closed_by_stop:
                self.upper_stop_closed_flag = True
            else:
                self.upper_stop_closed_flag = False
        elif pos["type"] == "lower":
            profit = price - pos["entry"]
            msg = (f"[Тейкпрофит/Стоплосс] {timestamp}: Цена = {price:.5f}.\n")
            if price <= self.indicators["lower_band_3"]:
                msg += f"Закрытие длинной позиции по стопу. Прибыль: {profit:.5f}"
                closed_by_stop = True
            else:
                msg += f"Закрытие длинной позиции по тейкпрофиту. Прибыль: {profit:.5f}"
            logger.info(msg)
            await send_telegram_message(msg, self.telegram_bot_token, self.telegram_chat_id)
            await self.send_market_order(self.symbol1, "SELL", pos["symbol1_quantity"])
            await self.send_market_order(self.symbol2, "BUY", pos["symbol2_quantity"])
            if closed_by_stop:
                self.lower_stop_closed_flag = True
            else:
                self.lower_stop_closed_flag = False
        self.positions.remove(pos)

    async def check_live_signals(self, current_price):
        async with self.signal_lock:
            if self.indicators is None:
                return

            ma = self.indicators["MA"]

            # Сначала проверяем условия закрытия для открытых позиций
            for pos in self.positions.copy():
                if pos["type"] == "upper":
                    if current_price <= ma or current_price >= self.indicators["upper_band_3"]:
                        await self.close_position(pos, current_price)
                elif pos["type"] == "lower":
                    if current_price >= ma or current_price <= self.indicators["lower_band_3"]:
                        await self.close_position(pos, current_price)

            # Для верхнего направления: если позиция закрыта по стопу, ждём обратного пересечения
            if self.upper_stop_closed_flag:
                if current_price < self.indicators["upper_band_2"]:
                    logger.info("Обратное пересечение для верхней стороны обнаружено, флаг стопа сброшен.")
                    self.upper_stop_closed_flag = False

            # Если флаг не установлен, проверяем условия открытия сделок для верхнего направления
            if not self.upper_stop_closed_flag:
                if current_price >= self.indicators["upper_band_2"] and not self.position_exists("upper", 2):
                    await self.open_position("upper", current_price, sigma_trigger=2)
                elif current_price >= self.indicators["upper_band_1"] and not self.position_exists("upper", 1):
                    await self.open_position("upper", current_price, sigma_trigger=1)

            # Аналогично для нижнего направления: ждём обратного пересечения, если стоп был сработан
            if self.lower_stop_closed_flag:
                if current_price > self.indicators["lower_band_2"]:
                    logger.info("Обратное пересечение для нижней стороны обнаружено, флаг стопа сброшен.")
                    self.lower_stop_closed_flag = False

            if not self.lower_stop_closed_flag:
                if current_price <= self.indicators["lower_band_2"] and not self.position_exists("lower", 2):
                    await self.open_position("lower", current_price, sigma_trigger=2)
                elif current_price <= self.indicators["lower_band_1"] and not self.position_exists("lower", 1):
                    await self.open_position("lower", current_price, sigma_trigger=1)


async def listen_trades(symbol, bot):
    stream_name = symbol.lower() + "@aggTrade"
    url = "wss://fstream.binance.com/ws"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                subscribe_msg = {"method": "SUBSCRIBE", "params": [stream_name], "id": 1}
                await ws.send_json(subscribe_msg)
                logger.info(f"Подписка на {stream_name} для {symbol} выполнена")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        if "p" in data:
                            price = float(data["p"])
                            await bot.update_price(symbol, price)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Ошибка в websocket для {symbol}: {msg.data}")
                        break
    except Exception as e:
        logger.error(f"Ошибка подключения к websocket для {symbol}: {e}")


async def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    ssl_context = ssl.create_default_context()
    client = await AsyncClient.create(
        api_key=config["api_key"],
        api_secret=config["api_secret"],
        tld='com',
        requests_params={'ssl': ssl_context}
    )

    bot = AsyncTradingBot(config, client)
    await bot.load_historical_data()

    tasks = [
        listen_trades(bot.symbol1, bot),
        listen_trades(bot.symbol2, bot)
    ]
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Ошибка в основной функции: {e}")
    finally:
        await client.close_connection()
        logger.info("Клиентская сессия Binance закрыта")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")

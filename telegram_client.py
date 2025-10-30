import asyncio
from telegram import Bot, error
from telegram.constants import ParseMode
from config import Config
from logger import logger


class TelegramClient:
    def __init__(self, token: str, channel_id: str):
        self.token = token
        self.channel_id = channel_id
        self.bot = Bot(token=self.token)

    async def send_signal(self, signal: dict) -> bool:
        """
        Sends formatted signal message to Telegram channel/group.
        Signal dict example:
        {
            "pair": "EUR/USD",
            "timeframe": "1 MIN",
            "signal": "CALL",
            "confidence": 0.95,
            "market_mood": "Bullish",
            "indicators": "RSI=58 | Cloud=Bullish | Bollinger=Breakout",
            "session": "London",
            "status": "ACTIVE",
            "strategy": "Sniper Reversal",
            "timestamp": "2024-06-01 12:34:56"
        }
        """
        try:
            message = self.format_message(signal)
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True,
            )
            logger.info(f"Sent signal to Telegram: {signal.get('pair', 'N/A')} Signal: {signal.get('signal', 'N/A')}")
            return True
        except error.TelegramError as e:
            logger.error(f"Telegram API error: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
        return False

    def format_message(self, signal: dict) -> str:
        """
        Formats signal dict into Telegram markdown message.
        Escapes special characters for MarkdownV2.
        """
        def escape_md(text: str) -> str:
            escape_chars = r'\_*[]()~`>#+-=|{}.!'
            return ''.join(f'\\{c}' if c in escape_chars else c for c in text)

        pair = escape_md(signal.get("pair", "N/A"))
        timeframe = escape_md(signal.get("timeframe", "N/A"))
        signal_val = escape_md(signal.get("signal", "N/A"))
        emoji = "🟢" if signal_val.upper() in ["CALL", "BUY"] else "🔴" if signal_val.upper() in ["PUT", "SELL"] else "⚪️"
        confidence = int(signal.get("confidence", 0) * 100)
        market_mood = escape_md(signal.get("market_mood", "Neutral"))
        indicators = escape_md(signal.get("indicators", "N/A"))
        session = escape_md(signal.get("session", "N/A"))
        status = escape_md(signal.get("status", "N/A"))
        strategy = escape_md(signal.get("strategy", "N/A"))
        timestamp = escape_md(signal.get("timestamp", "N/A"))

        msg = (
            f"*⚡️ MAYANK AI SIGNAL BOT ⚡️*\n\n"
            f"*Pair:* {pair}\n"
            f"*Timeframe:* {timeframe}\n"
            f"*Signal:* {signal_val} {emoji}\n"
            f"*Confidence:* {confidence}%\n"
            f"*Market Mood:* {market_mood} 🧠\n"
            f"*Indicators:* {indicators}\n"
            f"*Session:* {session} 🕐\n"
            f"*Status:* {status} ✅\n"
            f"*Strategy:* {strategy}\n"
            f"*Timestamp:* {timestamp}\n"
        )
        return msg


telegram_client = TelegramClient(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHANNEL_ID)

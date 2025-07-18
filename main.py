import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging

# Настройка логирования
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Токен бота из переменной окружения
TOKEN = os.getenv("BOT_TOKEN")

# Путь к файлам
DATA_FILE = "data.csv"
VECTORIZER_FILE = "vectorizer.pkl"
MODEL_FILE = "model.pkl"

# Инициализация или загрузка датасета
def init_dataset():
    if not os.path.exists(DATA_FILE):
        initial_data = pd.DataFrame({
            "text": ["лох", "пенис", "пидар", "гей", "penis", "piska", "shopa", "привет", "хороший день"],
            "label": [1, 1, 1, 1, 1, 1, 1, 0, 0]
        })
        initial_data.to_csv(DATA_FILE, index=False)
        logging.info("Создан начальный датасет")

# Загрузка или обучение модели
def load_or_train_model():
    init_dataset()
    try:
        data = pd.read_csv(DATA_FILE)
        texts = data["text"]
        labels = data["label"]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression()
        model.fit(X, labels)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        joblib.dump(model, MODEL_FILE)
        logging.info("Модель обучена и сохранена")
        return vectorizer, model
    except Exception as e:
        logging.error(f"Ошибка при обучении модели: {e}")
        raise

# Загрузка модели и векторизатора
try:
    vectorizer = joblib.load(VECTORIZER_FILE)
    model = joblib.load(MODEL_FILE)
    logging.info("Модель и векторизатор загружены")
except:
    vectorizer, model = load_or_train_model()

# Проверка плохих слов
def check_bad_words(text):
    try:
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0] == 1
        logging.info(f"Проверка текста '{text}': {'Плохое слово' if prediction else 'Нормальное слово'}")
        return prediction
    except Exception as e:
        logging.error(f"Ошибка при проверке текста: {e}")
        return False

# Добавление нового слова в датасет и переобучение
def add_bad_word_to_dataset(word):
    try:
        with open(DATA_FILE, "a") as f:
            f.write(f"{word},1\n")
        logging.info(f"Добавлено слово: {word}")
        # Переобучение модели
        global vectorizer, model
        vectorizer, model = load_or_train_model()
    except Exception as e:
        logging.error(f"Ошибка при добавлении слова: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот, который удаляет сообщения с нецензурными словами. Админы могут добавлять новые слова через /add_bad_word.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if check_bad_words(text):
        try:
            await update.message.delete()
            await update.message.reply_text("Сообщение удалено: использование нецензурных слов запрещено!")
            logging.info(f"Сообщение удалено: {text}")
        except Exception as e:
            await update.message.reply_text(f"Ошибка при удалении сообщения: {e}")
            logging.error(f"Ошибка при удалении сообщения: {e}")

async def add_bad_word(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    try:
        admins = await context.bot.get_chat_administrators(chat_id)
        if user_id not in [admin.user.id for admin in admins]:
            await update.message.reply_text("Только администраторы могут добавлять слова!")
            logging.warning(f"Пользователь {user_id} пытался добавить слово, но не является админом")
            return
        if not context.args:
            await update.message.reply_text("Укажите слово: /add_bad_word <слово>")
            return
        new_word = context.args[0]
        add_bad_word_to_dataset(new_word)
        await update.message.reply_text(f"Слово '{new_word}' добавлено в список плохих слов.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        logging.error(f"Ошибка при добавлении слова: {e}")

def main():
    try:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("add_bad_word", add_bad_word))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        logging.info("Бот запущен")
        app.run_polling()
    except Exception as e:
        logging.error(f"Ошибка при запуске бота: {e}")
        raise

if __name__ == '__main__':
    main()

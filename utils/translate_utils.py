from deep_translator import GoogleTranslator
from langdetect import detect


def detect_language(text):

    try:
        return detect(text)

    except:
        return "en"


def translate_to_english(text):

    try:
        return GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

    except:
        return text


def translate_from_english(
    text,
    target_lang
):

    try:
        return GoogleTranslator(
            source="en",
            target=target_lang
        ).translate(text)

    except:
        return text
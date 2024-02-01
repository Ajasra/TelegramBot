import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv

load_dotenv()
CR_DATABASE_URL = os.getenv("CR_DATABASE_URL")

logging = True

if logging:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, CR_DATABASE_URL)


@contextmanager
def get_db_cursor():
    conn = db_pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor
        conn.commit()
    except Exception as err:
        conn.rollback()
    finally:
        db_pool.putconn(conn)


def write_history_to_db(chat_id, prompt, answer, user_name="", user_id=0):
    """
    Write history to database
    :param user_id:
    :param user_name:
    :param chat_id:
    :param prompt:
    :param answer:
    :return:
    """
    if logging:
        with get_db_cursor() as cursor:
            if cursor:
                cursor.execute(
                    "INSERT INTO history (chat_id, question, answer, time, user_name, user_id) VALUES (%s, %s, %s, "
                    "NOW(), %s, %s)",
                    (chat_id, prompt, answer, user_name, user_id)
                )
                return cursor.rowcount == 1
        return False
    else:
        return True


def get_history_from_db(chat_id, limit=5):
    """
    Get history from database
    :param limit:
    :param chat_id:
    :return:
    """
    if logging:
        with get_db_cursor() as cursor:
            if cursor:
                cursor.execute(
                    "SELECT * FROM history WHERE chat_id = %s ORDER BY time DESC LIMIT %s",
                    (chat_id, limit)
                )
                return cursor.fetchall()
        return []
    else:
        return []
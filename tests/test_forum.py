# tests/test_forum.py

from web_raider.forums import Forum
import logging

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename='test_forum.log', level=logging.INFO)
    logging.info('Test Started')

    TEST_URL = "https://stackoverflow.com/questions/11875770/how-can-i-overcome-datetime-datetime-not-json-serializable?rq=2"
    TEST_ID = 11875813
    forum = Forum(TEST_URL)
    print(forum.get_answer_body(TEST_ID))
    ans = forum.parse_answer_body(forum.get_answer_body(TEST_ID))

    logger.info(type(ans))
    logger.info(ans)

    logger.info('Done')

if __name__ == '__main__':
    main()
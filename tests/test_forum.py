# tests/test_forum.py

from web_raider.forums import Forum
import logging

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename='test_forum.log', level=logging.INFO)
    logging.info('Test Started')

    TEST_URL = "https://stackoverflow.com/questions/11875770/how-can-i-overcome-datetime-datetime-not-json-serializable?rq=2"
    forum = Forum(TEST_URL)
    forum.get_answer_ids()
    logging.info(forum.answer_ids)
    forum.parse_all_answers('python datetime')

    logger.info(type(forum.tidied_ans))
    logger.info(forum.tidied_ans)

    logger.info('Done')

if __name__ == '__main__':
    main()
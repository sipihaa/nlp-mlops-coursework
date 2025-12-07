import os
import pandas as pd
import vk_api
from vk_api.exceptions import ApiError
import time
from dotenv import load_dotenv


load_dotenv()

aviation_groups = ['-18954214', '-44884514', '-79459310', '-23548502', '-57404545', '-78229688', '-43879844', '-145294610', '-116584424', '-188290249', '-156572447']
road_transport_groups = ['-120531646', '-78049396', '-56338600', '-184145186', '-19779149', '-226331383', '-45403349', '-219748240', '-133755929', '-152887367', '-129769303']

VK_API_TOKEN = os.getenv('VK_API_TOKEN')

vk_session = vk_api.VkApi(token=VK_API_TOKEN)
api = vk_session.get_api()


def parse_group_posts(group_id):
    posts = []
    offset = 0

    print(f"Начинаем парсинг группы {group_id}...")
    while True:
        try:
            # Запрос постов
            response = api.wall.get(
                owner_id=group_id, 
                offset=offset, 
                count=100
            )
            
            items = response['items']
            
            # Если посты кончились
            if not items:
                print("Посты закончились.")
                break
                
            # Сохраняем
            posts.extend(items)
            
            offset += 100
            
            # Визуализация прогресса
            print(f"Загружено: {len(posts)} постов")
            
            # ВАЖНО: Пауза, чтобы не ловить Error 29
            time.sleep(0.4) 

        except ApiError as e:
            if e.code == 29:
                print("Поймали Rate Limit (Err 29). Ждем 10 секунд...")
                time.sleep(10) # Ждем долго
                continue       # И пробуем ТОТ ЖЕ запрос снова (offset не меняем)
            else:
                print(f"Ошибка API Error: {e}")
                break
        
        except Exception as e:
            print(f"❌ Неизвестная ошибка: {e}")
            break

    print(f"\nИтого скачано: {len(posts)}")

    return posts


avia_posts = []
for ag in aviation_groups:
    avia_posts.extend(parse_group_posts(ag))
avia_data = pd.DataFrame(data = avia_posts)
avia_data['y'] = 'aviation'
avia_data.to_csv('data/raw/aviation.csv')

road_posts = []
for rtg in road_transport_groups:
    road_posts.extend(parse_group_posts(rtg))
road_data = pd.DataFrame(data = road_posts)
road_data['y'] = 'road_transport'
road_data.to_csv('data/raw/road_transport.csv')
import numpy as np

from Obstacle import *
from others import *
from Dino import *


# noinspection PyTypeChecker,PyGlobalUndefined,SpellCheckingInspection
def gameplay(dinoGroup):
    global high_score
    game_speed = 10
    startMenu = False
    gameOver = False
    gameQuit = False

    # 游戏参数初始化
    new_ground = Ground(-1 * game_speed)
    scb = Scoreboard()
    highsc = Scoreboard(width * 0.78)
    counter = 0
    cacti = pygame.sprite.Group()
    pteras = pygame.sprite.Group()
    clouds = pygame.sprite.Group()
    last_obstacle = pygame.sprite.Group()
    Cactus.containers = cacti
    Ptera.containers = pteras
    Cloud.containers = clouds

    temp_images, temp_rect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
    HI_image = pygame.Surface((22, int(11 * 6 / 5)))
    HI_rect = HI_image.get_rect()
    HI_image.fill(background_col)
    HI_image.blit(temp_images[10], temp_rect)
    temp_rect.left += temp_rect.width
    HI_image.blit(temp_images[11], temp_rect)
    HI_rect.top = height * 0.1
    HI_rect.left = width * 0.73

    population = len(dinoGroup.sprites())

    # 死亡的恐龙列表
    deadDino = []
    dinoDead = len(deadDino)
    dinoAlive = population - dinoDead

    # 默认第一只是玩家
    playerDino = dinoGroup.sprites()[0]

    while not gameQuit:
        while startMenu:
            pass
        while not gameOver:
            if pygame.display.get_surface() is None:
                print("Couldn't load display surface")
                gameQuit = True
                gameOver = True
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gameQuit = True
                        gameOver = True

            # 智能代码注入处

            print("\rdinoAlive: ", dinoAlive, end="")

            obstacle = []

            for c in cacti:
                obstacle.append(c)
            for p in pteras:
                obstacle.append(p)

            # 获取最近的障碍物的距离和索引
            obstacle_distance = np.inf
            obstacle_height = 0
            obstacle_width = 0
            if len(obstacle) > 0:
                obstacle.sort(key=lambda x: x.rect.left - playerDino.rect.right)

                # 获取最近的障碍物的距离
                obstacle_distance = obstacle[0].rect.left - playerDino.rect.right
                # 获取最近的障碍物离地面的高度
                obstacle_height = int(0.98 * height) - obstacle[0].rect.bottom
                # 获取最近的障碍物的宽度
                obstacle_width = obstacle[0].rect.width

            # 获得游戏速度
            dino_speed = game_speed

            # 将上面四个列表转换为numpy数组，要求每个列表只取第一个元素，生成的数组为1*4的数组
            isGaming = False
            info = np.array([0, 0, 0, 0])
            try:
                info = np.array([obstacle_distance, obstacle_height, dino_speed, obstacle_width])
                print("\tinfo: ", info, end="")
                isGaming = True
            except IndexError:
                isGaming = False

            if isGaming:
                for i, dino in enumerate(dinoGroup):
                    if info[0] == np.inf:
                        pass
                        continue

                    # 对info进行标准化
                    info = (info - info.mean()) / info.std()

                    # 用神经网络预测
                    prediction = dino.NN.predict(info)
                    # 选择最大的那个动作
                    action = np.argmax(prediction)
                    # 执行动作
                    if action == 0:
                        dino.jump()
                    elif action == 1:
                        pass
                    elif action == 2:
                        dino.duck()

            # 对每只恐龙进行碰撞检测
            for dino in dinoGroup:
                for c in cacti:
                    c.movement[0] = -1 * game_speed
                    if pygame.sprite.collide_mask(dino, c):
                        dino.isDead = True
                        # if pygame.mixer.get_init() is not None:
                        #     die_sound.play()
                for p in pteras:
                    p.movement[0] = -1 * game_speed
                    if pygame.sprite.collide_mask(dino, p):
                        dino.isDead = True
                        # if pygame.mixer.get_init() is not None:
                        #     die_sound.play()

            # 生成障碍物
            if len(cacti) < 2:
                if len(last_obstacle) == 0:
                    last_obstacle.empty()
                    last_obstacle.add(Cactus(game_speed, 40, 40))
                else:
                    for l in last_obstacle:
                        if l.rect.right < width * 0.6 and random.randrange(0, 50) == 10:
                            last_obstacle.empty()
                            last_obstacle.add(Cactus(game_speed, 40, 40))

            if len(pteras) == 0 and random.randrange(0, 200) == 10 and counter > 500:
                for l in last_obstacle:
                    if l.rect.right < width * 0.6:
                        last_obstacle.empty()
                        last_obstacle.add(Ptera(game_speed, 46, 40))

            if len(clouds) < 5 and random.randrange(0, 300) == 10:
                Cloud(width, random.randrange(height / 5, height / 2))

            score = playerDino.score
            # 如果恐龙死亡，删除恐龙
            for dino in dinoGroup:
                if dino.isDead:
                    deadDino.append(dino)
                    dinoGroup.remove(dino)

            dinoDead = len(deadDino)
            dinoAlive = population - dinoDead

            # 更新所有的恐龙
            dinoGroup.update()

            cacti.update()
            pteras.update()
            clouds.update()
            new_ground.update()
            scb.update(score)
            highsc.update(high_score)

            if pygame.display.get_surface() is not None:
                screen.fill(background_col)
                new_ground.draw()
                clouds.draw(screen)
                scb.draw()
                if high_score != 0:
                    highsc.draw()
                    screen.blit(HI_image, HI_rect)
                cacti.draw(screen)
                pteras.draw(screen)

                dinoGroup.draw(screen)
                pygame.display.update()
            clock.tick(FPS)

            # 检测死亡恐龙数量，如果全部死亡，则游戏结束
            if dinoAlive == 0:
                gameOver = True

                # 提取出分数
                scores = []
                for dino in deadDino:
                    scores.append(dino.score)

                def softmax(fitness_vector):
                    """
                    information: softmax函数将适应度向量转化为概率分布
                    creating time: 2023/3/16
                    """
                    temp = np.array(fitness_vector, dtype=np.float64)
                    return np.exp(temp) / np.exp(temp).sum()

                print("\nscores: ", scores)

                # 归一化函数
                def normalize(x):
                    # 判断是否全部一样
                    if x.max() == x.min():
                        x_norm = np.ones_like(x)
                    else:

                        x_norm = (x - x.min()) / (x.max() - x.min())
                    return x_norm

                scores = np.array(scores, dtype=np.float64)
                # 精英数量
                elite_size = 20
                # 将scores后面10个设为精英，单独提出来
                strong_individuals = scores[-elite_size:]
                normal_individuals = scores[:-elite_size]
                prob = np.concatenate(
                    (softmax(normalize(normal_individuals)) * 0.5, softmax(normalize(strong_individuals)) * 0.5))
                print("prob: ", prob)
                print("fitness: ", prob[:10], sum(prob[:10]))
                select_id = np.random.choice(range(population), 2 * population, replace=True,
                                             p=prob)  # replace=True说明可以有重复项，p=prob表示产生的随机选择要符合prob的概率分布
                print("select_id: ", select_id)
                selection = []
                for idx in select_id:
                    deadDino[idx].NN.fitness = prob[idx]
                    selection.append(deadDino[idx])

                children = []
                # 交叉
                for i in range(population):
                    father = selection[i].NN
                    mother = selection[population - 1 - i].NN
                    children.append(father.cross(mother))

                # 变异
                for i in range(population):
                    children[i].mutation()

                current_high_score = 0
                for dino in deadDino:
                    current_high_score = max(current_high_score, dino.score)
                high_score = max(high_score, current_high_score)

                elite_saved = 1
                l = 0.05
                old = int(population * l)
                new = population - old - elite_saved
                # 将恐龙的神经网络替换为新的
                for i in range(old):
                    dinoGroup.add(Dino(44, 47, children[i]))
                for i in range(new):
                    dinoGroup.add(Dino(44, 47))
                for i in range(elite_saved):
                    dinoGroup.add(Dino(44, 47, deadDino[-i - 1].NN))
                print("current_high_score: ", current_high_score)
                print("high_score: ", high_score)
                return dinoGroup
            else:
                playerDino = dinoGroup.sprites()[0]
            if counter % 700 == 699:
                new_ground.speed -= 1
                if game_speed < 21:
                    game_speed += 1
            counter = (counter + 1)
        if gameQuit:
            break
    pygame.quit()
    quit()


def main():
    # 加一群恐龙
    dinoGroup = pygame.sprite.Group()
    Dino.containers = dinoGroup
    population = 1000
    generation = 1000
    for i in range(population):
        dinoGroup.add(Dino(44, 47))
    for i in range(generation):
        print()
        print("generation: ", i + 1)
        dinoGroup = gameplay(dinoGroup)
    # 将最后一代的最后一个恐龙的神经网络保存下来
    with open("dino.pkl", "wb") as f:
        dinoGroup.sprites()[-1].NN.save(f)

    sys.exit()


if __name__ == '__main__':
    main()

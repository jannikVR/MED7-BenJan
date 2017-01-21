import pygame
import numpy as np
import math
import csv


class PerformanceView():

    def __init__(self,screenSize, distFromEdge = 50, caption = "Performance", bgCol = (255,255,255), frontCol = (10,50,140)):

        pygame.init()
        self.__graphCol = ([[220,40,40],[220,220,40],[40,220,40],[40,40,220],[40,220,220],[220,40,220],[40,40,40]])
                            #[150, 20, 20], [150, 150, 20], [20, 150, 20], [20, 20, 150], [20, 150, 150], [150, 20, 150],
                            #[70, 70, 70]])
        pygame.display.set_caption(caption)
        self.__distFromEdge = distFromEdge
        self.__settingID = -1
        self.__performances = []
        self.__rewards = []
        self.__error = []
        self.__performanceSettings = []
        self.__performancesSaved = -1
        self.__bgCol = bgCol
        self.__frontCol = frontCol
        self.__screenSize = screenSize
        self.__screen = pygame.display.set_mode(screenSize)
        self.__frameRate = 30
        self.__clock = pygame.time.Clock()

        self.__font = pygame.font.Font(None,25)

        #self.__performances = np.zeros((attemptsPrSetting,maxEpisodes,numberOfSettings))

        # for i in range(maxEpisodes):
        #     self.__performances.append([])
        #     for j in range(numberOfSettings):
        #         self.__performances[i].append(0)

    def update(self,maxEpisodes,maxFreq, maxReward, rewardResolution, settingAttempt, attemptsPrSetting):

        graphW = self.__screenSize[0]-self.__distFromEdge*2
        graphH = self.__screenSize[1]-self.__distFromEdge*5
        barWidth = 3
        spaceBetweenBars = 6
        binSize = 100
        totalBins = int(maxEpisodes/binSize)+1
        binData = []


        self.__screen.fill(self.__bgCol)

        legendName = self.__font.render("Hidden Layers || Neurons Pr. Layer ||       Alpha       || Alpha Decrease || "
        #                               "Discount factor || A Decrease Type", True, [0,0,0])
        #legendName = self.__font.render("       Alpha       || Alpha Decrease || "
                                        "Discount factor || A Decrease Type", True, [0, 0, 0])

        offset = 0
        self.__screen.blit(legendName, [self.__distFromEdge, self.__distFromEdge-25])

        for i in range(self.__performancesSaved+1):
            binData.append([0] * totalBins)

            for x in range(offset,len(self.__performanceSettings[0])):
                legendText = self.__font.render(str(self.__performanceSettings[i][x]), True,self.__graphCol[i])
                self.__screen.blit(legendText, [50+self.__distFromEdge+140*(x-offset), self.__distFromEdge + 25 * i])

            for j in range(len(self.__performances[i])):
                binData[i][math.floor(self.__performances[i][j]/binSize)] += 1

        #Reward graph
        settingAttempt = max(settingAttempt, 1)
        print(settingAttempt)

        pygame.draw.line(self.__screen,(0,120,30),(self.__distFromEdge, self.__screenSize[1] - self.__distFromEdge - graphH
                         * (195 / maxReward)),(self.__distFromEdge +graphW, self.__screenSize[1] - self.__distFromEdge - graphH
                         * (195 / maxReward)),2)

        pygame.draw.line(self.__screen, (0, 0, 0),
                         (self.__distFromEdge, self.__screenSize[1] - self.__distFromEdge - graphH
                          * (100 / maxReward)),
                         (self.__distFromEdge + graphW, self.__screenSize[1] - self.__distFromEdge - graphH
                          * (100 / maxReward)))

        for subAttempt in range(len(self.__rewards)):
            #print("Lenght of reward list: ", len(self.__rewards[subAttempt]) - 1)

            for subBin in range(len(self.__rewards[subAttempt]) - 1):
                pygame.draw.line(self.__screen, self.__graphCol[math.floor(subAttempt/attemptsPrSetting)],
                                 (self.__distFromEdge + graphW * (subBin / rewardResolution),
                                  self.__screenSize[1] - self.__distFromEdge - graphH * (
                                  self.__rewards[subAttempt][subBin] / maxReward)),
                                 (self.__distFromEdge + graphW * ((subBin + 1) / rewardResolution),
                                  self.__screenSize[1] - self.__distFromEdge - graphH * (
                                  self.__rewards[subAttempt][subBin + 1] / maxReward)), 2)

                # pygame.draw.line(self.__screen, (self.__graphCol[math.floor(subAttempt / attemptsPrSetting)][0] - 40,
                #                                  self.__graphCol[math.floor(subAttempt / attemptsPrSetting)][1]-40,
                #                                  self.__graphCol[math.floor(subAttempt / attemptsPrSetting)][2]-40),
                #                  (self.__distFromEdge + graphW * (subBin / rewardResolution),
                #                   self.__screenSize[1] - self.__distFromEdge - graphH * (
                #                       self.__error[subAttempt][subBin] / 1)),
                #                  (self.__distFromEdge + graphW * ((subBin + 1) / rewardResolution),
                #                   self.__screenSize[1] - self.__distFromEdge - graphH * (
                #                       self.__error[subAttempt][subBin + 1] / 1)), 2)


        for attempt in range(self.__performancesSaved+1):

            binWidth = (graphW / totalBins)

            pygame.draw.line(self.__screen, (0, 0, 0),
                             (self.__distFromEdge + (totalBins+1) * binWidth, self.__screenSize[1] - self.__distFromEdge - graphH),
                             (self.__distFromEdge + (totalBins+1) * binWidth, self.__screenSize[1] - self.__distFromEdge
                              ))

            for bin in range(totalBins):

                baseX = self.__distFromEdge + bin * binWidth

                pygame.draw.line(self.__screen, (0, 0, 0),
                                 (baseX, self.__screenSize[1] - self.__distFromEdge - graphH),
                                 (baseX, self.__screenSize[1] - self.__distFromEdge
                                    ))


                if attempt == 0:
                    text = self.__font.render(str(binSize * bin), True, self.__graphCol[self.__performancesSaved])
                    self.__screen.blit(text, [baseX,
                                              self.__screenSize[1] - self.__distFromEdge + 25])

                if binData[attempt][bin] > 0:
                    pygame.draw.rect(self.__screen, self.__graphCol[attempt], [baseX+(spaceBetweenBars*attempt),
                                self.__screenSize[1] - self.__distFromEdge, barWidth, -binData[attempt][bin]/maxFreq*graphH])
                    text = self.__font.render(str(binData[attempt][bin]), True, (0,0,0))
                    self.__screen.blit(text, [baseX+(spaceBetweenBars*attempt),
                                              ((self.__screenSize[1] - self.__distFromEdge) - binData[attempt][bin] / maxFreq * graphH) - 25])

            # pygame.draw.rect(self.__screen, self.__frontCol,
            #                  [self.__distFromEdge + barWidth * j, self.__screenSize[1] - self.__distFromEdge, 10,
            #                   -self.__performances[i][j] / maxEpisodes * graphH])

        pygame.display.flip()

    def addPerformance(self, settingID, episodesToWin, settings, reward,attempt, totalAttempts, error):

        if self.__settingID != settingID:
            self.__settingID = settingID
            self.__performancesSaved += 1
            self.__performances.append([])
            self.__performanceSettings.append(settings)

        self.__rewards.append([])
        self.__error.append([])
        #self.addReward(reward,attempt+(self.__performancesSaved*totalAttempts))
        self.addReward(reward, attempt)
        #self.addError(error, attempt)

        self.__performances[self.__performancesSaved].append(episodesToWin)

    def addReward(self, reward,attempt):

        for r in reward:
            self.__rewards[attempt].append(r)

    def addError(self, error,attempt):

        for e in error:
            self.__error[attempt].append(e)


    def saveScreenshot(self, filename):
        pygame.image.save(self.__screen, filename+".png")
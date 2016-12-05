import pygame
import numpy as np
import math


class PerformanceView():

    def __init__(self,screenSize, distFromEdge = 50, caption = "Performance", bgCol = (255,255,255), frontCol = (10,50,140)):

        pygame.init()
        self.__graphCol = ([[220,20,20],[220,220,20],[20,220,20],[20,20,220],[20,220,220],[220,20,220],[20,20,20]])
                            #[150, 20, 20], [150, 150, 20], [20, 150, 20], [20, 20, 150], [20, 150, 150], [150, 20, 150],
                            #[70, 70, 70]])
        pygame.display.set_caption(caption)
        self.__distFromEdge = distFromEdge
        self.__settingID = -1
        self.__performances = []
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

    def update(self,maxEpisodes,maxFreq):

        graphW = self.__screenSize[0]-self.__distFromEdge*2
        graphH = self.__screenSize[1]-self.__distFromEdge*5
        barWidth = 3
        spaceBetweenBars = 6
        binSize = 100
        totalBins = int(maxEpisodes/binSize)+1
        binData = []


        self.__screen.fill(self.__bgCol)

        legendName = self.__font.render("Hidden Layers, Neurons Pr. Layer, Alpha, Alpha Decrease", True, [0,0,0])
        self.__screen.blit(legendName, [self.__distFromEdge, self.__distFromEdge-25])

        for i in range(self.__performancesSaved+1):
            binData.append([0] * totalBins)

            for x in range(len(self.__performanceSettings[0])):
                legendText = self.__font.render(str(self.__performanceSettings[i][x]), True,self.__graphCol[i])
                self.__screen.blit(legendText, [self.__distFromEdge+125*x, self.__distFromEdge + 25 * i])

            for j in range(len(self.__performances[i])):
                binData[i][math.floor(self.__performances[i][j]/binSize)] += 1

        print("BIN DATA ##### ::: ",binData)

        for attempt in range(self.__performancesSaved+1):
            for bin in range(totalBins):

                if attempt == 0:
                    text = self.__font.render(str(binSize * bin), True, self.__graphCol[self.__performancesSaved])
                    self.__screen.blit(text, [self.__distFromEdge + bin * (graphW / totalBins),
                                              self.__screenSize[1] - self.__distFromEdge + 25])

                if binData[attempt][bin] > 0:
                    pygame.draw.rect(self.__screen, self.__graphCol[attempt], [self.__distFromEdge+bin*(graphW/totalBins)+(spaceBetweenBars*attempt),
                                self.__screenSize[1] - self.__distFromEdge, barWidth, -binData[attempt][bin]/maxFreq*graphH])
                    text = self.__font.render(str(binData[attempt][bin]), True, self.__graphCol[attempt])
                    self.__screen.blit(text, [self.__distFromEdge+bin*(graphW/totalBins)+(spaceBetweenBars*attempt),
                                              ((self.__screenSize[1] - self.__distFromEdge) - binData[attempt][bin] / maxFreq * graphH) - 25])

            # pygame.draw.rect(self.__screen, self.__frontCol,
            #                  [self.__distFromEdge + barWidth * j, self.__screenSize[1] - self.__distFromEdge, 10,
            #                   -self.__performances[i][j] / maxEpisodes * graphH])

        pygame.display.flip()

    def addPerformance(self, settingID, episodesToWin, settings):

        if self.__settingID != settingID:
            self.__settingID = settingID
            self.__performancesSaved += 1
            self.__performances.append([])
            self.__performanceSettings.append(settings)

        self.__performances[self.__performancesSaved].append(episodesToWin)

    def saveScreenshot(self, filename):
        pygame.image.save(self.__screen, filename+".png")
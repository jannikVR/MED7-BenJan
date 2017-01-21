import pygame
import numpy as np
import random



class Gridworld():


    def __init__(self, tileSize, envDim, startTile, endTiles, trapTiles, goalReward = 1, trapReward = -1, standardReward = 0):

        pygame.init()

        #env setup
        self.__sqSize = tileSize
        self.__envDim = envDim
        self.__startTile = startTile
        self.__endTiles = endTiles       # only one now
        self.__trapTiles = trapTiles     # only more than one now


        #colours
        self.__backgroundC = (255, 255, 255)
        self.__lineC = (10, 10, 10)

        #setup window
        self.__buttomMargen = 100
        self.__screensize = self.__sqSize * self.__envDim[0], self.__sqSize * self.__envDim[1] + self.__buttomMargen
        self.__window = pygame.display.set_mode(self.__screensize)
        pygame.display.set_caption("gridworld")
        pygame.font.Font(None, 25)
        self.__window.fill(self.__backgroundC)

        # rewards setup
        self.__goalR = goalReward
        self.__trapR = trapReward
        self.__standardR = standardReward

        # saved player data
        self.__playerPos = self.__startTile

    def performAction(self, currentPos, actionNr): # 0 = L,  1 = U, 2 = R, 3 = D

        repositioned = False
        #new position
        newPos = currentPos
        if actionNr == 0: # L
            if currentPos[0] - 1 >= 0:
                newPos = (currentPos[0] - 1, currentPos[1])

        if actionNr == 1: # U
            if currentPos[1] - 1 >= 0:
                newPos = (currentPos[0], currentPos[1] - 1)

        if actionNr == 2: # R
            if currentPos[0] + 1 < envDim[0]:
                newPos = (currentPos[0] + 1, currentPos[1])

        if actionNr == 3: # D
            if currentPos[1] + 1 < envDim[1]:
                newPos = (currentPos[0], currentPos[1] + 1)

        #reward
        reward = self.__getReward(newPos)
        if reward != self.__standardR:     # if hit trap or goal
            newPos = self.__startTile      # move to start
            repositioned = True

        self.__playerPos = newPos

        return newPos, reward, repositioned

    def startUpdating(self):                            # before uodating q-values
        self.__window.fill(self.__backgroundC)

    def update(self, x, y, qvalues):

        self.__drawState(x * self.__sqSize, y * self.__sqSize, qvalues, self.__sqSize)

    def doneUpdating(self):                             # after all q-values are updated
        self.__drawGrid()
        self.__drawStandardTiles()
        self.__drawMyPos(self.__playerPos)
        pygame.display.flip()

    def __getReward(self, tilePosition):
        # check if position = goal/trap/blank
        if tilePosition == endTile:
            return self.__goalR

        for i in range(len(trapTiles)):
            if tilePosition == trapTiles[i]:
                return self.__trapR

        return self.__standardR

    def __drawState(self, posX, posY, qvalues, size):

        # calc triangle points
        pS = size  # point scaler
        pPX = posX  # point pos
        pPY = posY

        plL = [(0 + pPX, 0 + pPY),
               (0.5 * pS + pPX, 0.5 * pS + pPY),
               (0 + pPX, 1 * pS + pPY),
               (0 + pPX, 0 + pPY)]
        plU = [(0 + pPX, 0 + pPY),
               (0.5 * pS + pPX, 0.5 * pS + pPY),
               (1 * pS + pPX, 0 + pPY),
               (0 + pPX, 0 + pPY)]
        plR = [(1 * pS + pPX, 1 * pS + pPY),
               (0.5 * pS + pPX, 0.5 * pS + pPY),
               (1 * pS + pPX, 0 + pPY),
               (1 * pS + pPX, 1 * pS + pPY)]
        plD = [(1 * pS + pPX, 1 * pS + pPY),
               (0.5 * pS + pPX, 0.5 * pS + pPY),
               (0 + pPX, 1 * pS + pPY),
               (1 * pS + pPX, 1 * pS + pPY)]

        # draw triangles
        if qvalues[0] > 0:
            greenval = 255*qvalues[0]
            pygame.draw.polygon(self.__window, (255 - greenval, 255, 255 - greenval), plL, 0)
        else:
            redval = 255 * qvalues[0]*-1
            pygame.draw.polygon(self.__window, (255, 255 - redval, 255 - redval), plL, 0)

        if qvalues[1] > 0:
            greenval = 255*qvalues[1]
            pygame.draw.polygon(self.__window, (255 - greenval, 255, 255 - greenval), plU, 0)
        else:
            redval = 255 * qvalues[1]*-1
            pygame.draw.polygon(self.__window, (255, 255 - redval, 255 - redval), plU, 0)

        if qvalues[2] > 0:
            greenval = 255 * qvalues[2]
            pygame.draw.polygon(self.__window, (255 - greenval, 255, 255 - greenval), plR, 0)
        else:
            redval = 255 * qvalues[2] * -1
            pygame.draw.polygon(self.__window, (255, 255 - redval, 255 - redval), plR, 0)

        if qvalues[3] > 0:
            greenval = 255 * qvalues[3]
            pygame.draw.polygon(self.__window, (255 - greenval, 255, 255 - greenval), plD, 0)
        else:
            redval = 255 * qvalues[3] * -1
            pygame.draw.polygon(self.__window, (255, 255 - redval, 255 - redval), plD, 0)

        # calc and draw average square
        averageval =  (qvalues[0] + qvalues[1] + qvalues[2] + qvalues[3])/4


        if averageval > 0:
            averageval *= 255  # into color
            col = (255-averageval, 255, 255-averageval)
        else:
            averageval *= 255 * -1  # into color
            col = (255, 255 - averageval, 255 - averageval)
        sn = 0.5 # square size number
        pygame.draw.rect(self.__window, col, (posX + size * 0.5 - size * sn * 0.5, posY + size * 0.5 - size * sn * 0.5, size * sn, size * sn), 0)

    def __drawStandardTiles(self):
        # standard tile properties
        startTileCol = (255, 255, 0)
        endTileCol = (255, 0, 255)
        trapTileCol = (0, 0, 0)
        width = int(self.__sqSize / 5)

        # draw standard tiles start/end

        pos = self.__startTile
        pygame.draw.rect(self.__window, startTileCol, (pos[0] * self.__sqSize, pos[1] * self.__sqSize, self.__sqSize, self.__sqSize),
                         width)

        # for i in range(len(self.endTiles)):   #needs to be np object otherwise it can't see if (1,2) or ((1,2),(1,2))
        #     pos = self.endTiles[i]
        pos = self.__endTiles
        pygame.draw.rect(self.__window, endTileCol, (pos[0] * self.__sqSize, pos[1] * self.__sqSize, self.__sqSize, self.__sqSize),
                         width)

        # trap tiles
        for i in range(len(self.__trapTiles)):
            pos = self.__trapTiles[i]
            pygame.draw.rect(self.__window, trapTileCol,
                             (pos[0] * self.__sqSize, pos[1] * self.__sqSize, self.__sqSize, self.__sqSize), width)

    def __drawGrid(self):
        color = (180,180,180)
        width = int(self.__sqSize / 30)

        for x in range(self.__envDim[0]):
            pygame.draw.line(self.__window, color, (x * self.__sqSize, 0), (x * self.__sqSize, envDim[1] * self.__sqSize),
                             width)
        for y in range(self.__envDim[1]):
            pygame.draw.line(self.__window, color, (0, y * self.__sqSize), (envDim[0] * self.__sqSize, y * self.__sqSize),
                             width)

    def __drawMyPos(self, position):

        size = self.__sqSize
        col1 = (0, 0, 255)
        col2 = (80, 80, 255)
        col3 = (150, 150, 255)

        pos = int(position[0] * self.__sqSize + 0.5 * size), int(position[1] * self.__sqSize + 0.5 * size)

        sn = 0.5  # square size number
        pygame.draw.circle(self.__window, col1, pos, int(size * sn), 0)

        sn = 0.375  # square size number
        pygame.draw.circle(self.__window, col2, pos, int(size * sn), 0)

        sn = 0.2  # square size number
        pygame.draw.circle(self.__window, col3, pos, int(size * sn), 0)

#--------------------------------------------


# env setup
tileSize = 20
envDim = 35, 21
startTile = 0, int((envDim[1]-1)/2)          # auto
endTile = envDim[0]-1, int((envDim[1]-1)/2)  # auto
trapTiles = [(3,1), (4,5), (9,0)]

gworld = Gridworld(tileSize, envDim, startTile, endTile, trapTiles)


active = True
updateEnv = True
while active:

    # if QUIT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            active = False


    # update Env
    if updateEnv:
        # gworld.performAction((envDim[0] - 2, int((envDim[1] - 1) / 2)), 0)  # L
        gworld.startUpdating()                       # start updating

        for x in range(envDim[0]):                   # for each state
            for y in range(envDim[1]):
                # some random q-values
                val0 = random.random() * 2 - 1       # calculate q-values
                val1 = random.random() * 2 - 1
                val2 = random.random() * 2 - 1
                val3 = random.random() * 2 - 1
                qvalues = (val0, val1, val2, val3)

                gworld.update(x, y, qvalues)        # update

        gworld.doneUpdating()                       # stop updating


        # test
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 0)) # L
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 1)) # U
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 2)) # R
        # print(gworld.performAction((envDim[0]-2, int((envDim[1]-1)/2)), 3)) # D




        updateEnv = False





pygame.quit() # clears data pieces
quit()
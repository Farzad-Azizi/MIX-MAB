""" LPWAN Simulator: Hepper functions
============================================
Utilities (:mod:`lora.bsFunctions`)
============================================
.. autosummary::
   :toctree: generated/
   transmitPacket           -- Transmission process with discret event simulation.
   cuckooClock              -- Notify the simulation time (for each 1k hours).
   saveProb                 -- Save the probability profile of each node.
"""    
import os
import random
import numpy as np
from os.path import join
from .loratools import airtime, dBmTomW
# Transmit
def transmitPacket(env, node, bsDict, logDistParams, algo): 
    """ Transmit a packet from node to all BSs in the list.
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    node: my Node
        LoRa node.
    bsDict: dict
        list of BSs.
    logDistParams: list
        channel params
    algo: string
        learning algorithm
    Returns
    -------
    """
     
    while True:
        # The inter-packet waiting time. Assumed to be exponential here.
        yield env.timeout(random.expovariate(1/float(node.period)))
        
        # update settings if any
        node.updateTXSettings()
        node.resetACK()
        node.packetNumber += 1
        # send a virtual packet to each base-station in range and those we may affect
        for bsid, dist in node.proximateBS.items():
            
            prob_temp = [node.prob[x] for x in node.prob]
            npacketsSuccessful_temp = [node.npacketsSuccessful[x] for x in node.npacketsSuccessful]
            npacketsTransmitted_temp = [node.npacketsTransmitted[x] for x in node.npacketsTransmitted]
            node.packets[bsid].updateTXSettings(bsDict, logDistParams, prob_temp,npacketsTransmitted_temp,npacketsSuccessful_temp)
            bsDict[bsid].addPacket(node.nodeid, node.packets[bsid])
            bsDict[bsid].resetACK()
        

        # wait until critical section starts
        Tcritical = (2**node.packets[0].sf/node.packets[0].bw)*(node.packets[0].preambleLength - 5) # time until the start of the critical section
        yield env.timeout(Tcritical)
        
        # make the packet critical on all nearby basestations
        for bsid in node.proximateBS.keys():
            bsDict[bsid].makeCritical(node.nodeid)
            
        Trest = airtime((node.packets[0].sf, node.packets[0].rdd, node.packets[0].bw, node.packets[0].packetLength, node.packets[0].preambleLength, node.packets[0].syncLength, node.packets[0].headerEnable, node.packets[0].crc)) - Tcritical # time until the rest of the message completes
        yield env.timeout(Trest)
        
        successfulRx = False
        ACKrest = 0
        
        # transmit ACK
        for bsid in node.proximateBS.keys():
            if bsDict[bsid].removePacket(node.nodeid):
                bsDict[bsid].addACK(node.nodeid, node.packets[bsid])
                ACKrest = airtime((node.packets[0].sf, node.packets[0].rdd, node.packets[0].bw, node.packets[0].packetLength, node.packets[0].preambleLength, node.packets[0].syncLength, node.packets[0].headerEnable, node.packets[0].crc))# time until the ACK completes
                yield env.timeout(ACKrest)
                node.addACK(bsDict[bsid].bsid, node.packets[bsid])
                successfulRx = True
                
        # update probability        
        node.packetsTransmitted += 1 
        node.energy += node.packets[0].rectime * dBmTomW(node.packets[0].pTX) * (3.0) /1e6 # V = 3.0     # voltage XXX
        if successfulRx:
            if node.info_mode in ["NO", "PARTIAL"]:
                node.packetsSuccessful += 1
                node.transmitTime += node.packets[0].rectime 
            elif node.info_mode == "FULL": 
                if not node.ack[0].isCollision:
                    node.packetsSuccessful += 1
                    node.transmitTime += node.packets[0].rectime 

            node.updateProb(algo,node.packetsTransmitted,node.packetsSuccessful,node.nodeid)
        else:
             node.updateProb(algo,node.packetsTransmitted,node.packetsSuccessful,node.nodeid)

        yield env.timeout(float(node.period)-Tcritical-Trest-ACKrest)


def cuckooClock(env):
    """ Notifies the simulation time.
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    Returns
    -------
    """
    while True:
        #print (format(env.now))
        yield env.timeout(1000 * 3600000)
        print("Running {} kHrs".format(env.now/(1000 * 3600000)))


def saveRatio(env, nodeDict, fname, simu_dir):
    """ Save packet reception ratio to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(1000 * 3600000)
        # write packet reception ratio to file
        nTransmitted = 0
        nRecvd = 0
        PacketReceptionRatio = 0
        nTransmitted = sum(nodeDict[nodeid].packetsTransmitted for nodeid in nodeDict.keys())
        nRecvd = sum(nodeDict[nodeid].packetsSuccessful for nodeid in nodeDict.keys())
        PacketReceptionRatio = nRecvd/nTransmitted
        filename = join(simu_dir, str('Result'+ fname) + '.csv')
        if os.path.isfile(filename):
            res = "\n" + str(PacketReceptionRatio) + " " + format(env.now)+ " " + str(sum(node.energy for nodeid, node in nodeDict.items())/nTransmitted)
        else:
            res = str(PacketReceptionRatio) + " " + format(env.now) + " " + str(sum(node.energy for nodeid, node in nodeDict.items())/nTransmitted)
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()



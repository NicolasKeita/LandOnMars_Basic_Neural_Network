/*
    Path + Filename: src/index.js.ts
*/

import Point from './Point'
import {mars} from './dataPoints'

const landingArea = mars.find(({y}, i) => y === mars[i + 1]?.y)
if (!landingArea) throw new Error('ERR: mars is not flat')

console.log('helow world2')

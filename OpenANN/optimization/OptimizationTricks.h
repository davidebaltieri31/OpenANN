#pragma once


namespace OpenANN
{

/**
 * Evolutionary algorithms often do not consider any boundaries for parameters.
 * This function maps an optimized parameter to [0, max].
 * @param x optimized value
 * @param max maximal value
 * @return mapping from x to [0, max]
 */
fpt periodicMapping(fpt x, fpt max);

}

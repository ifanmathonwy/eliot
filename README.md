# Eliot

## About 

Eliot is a library for poetry generation. It works by generating words from a
language model, constrained by another model of the scansion and rhyme
proper to a particular poetic form.

The language model is a bigram model from NLTK and the poetry model uses 
regular expressions to match appropriate stress sequences.

## Examples

This is generated using a bigram language model derived from the writings of
William Blake:

> became a holy light the meanest thing
>
> delights in coffins of a falling tear
>
> disputes and happy sleep the green and sing
>
> delights in morning sky the humid air
>
> arise in one in heaven of delight
>
> around the sky the maid and boys and your
>
> er heaven of the father see a sight
>
> around the face a hell in sorrow sore
>
> away the voices of a rural pen
>
> in love and shine in sorrow pale and o
>
> upon the humble sheep a song again
>
> deceiving tears away the face a glow
>
> beside the secret love a tangle spray
>
> delights in hell in hell in senseless clay

And here are two limericks generated from the works of G. K. Chesterton.

The first:
> minority of a position
>
> examining it a commission
>
> essentially an
>
> presentable man
>
> agility hardly patrician

The second:
> explaining away a delusion
>
> hegemony quite in allusion
>
> electrical chair
>
> entirely their
>
> protection against the conclusion
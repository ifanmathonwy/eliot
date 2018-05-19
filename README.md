# Eliot

## About 

Eliot is a tool for poetry generation. It works by generating words from a
language model, constrained by another model of the scansion and rhyme
proper to a particular poetic form.

Currently it generates Shakespearian sonnets only. The language model is a
bigram model from NLTK and the poetry model uses regular expressions to
match appropriate stress sequences.

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
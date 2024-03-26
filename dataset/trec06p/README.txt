TREC 2006 Spam Track Public Corpus

----

INSTRUCTIONS

1. The compressed file may be uncompressed with gzip, Winzip,
   or any other utility that understands gzip format.

2. The compressed file will unpack to a folder named trec06

3. There is one main corpus with four subsets:

   trec06/full   -- the main corpus with messages (37822
                    messages; 12910 ham, 24912 spam)
   trec06/ham25  -- subset of full: 100% of spam, 25% of ham
   trec06/ham50  -- subset of full: 100% of spam, 50% of ham
   trec06/spam25 -- subset of full: 25% of spam, 100% of ham
   trec06/spam50 -- subset of full: 50% of spam, 100% of ham

4. There is a delayed-feedback version of each:

      run.sh trec06/full-delay/
      run.sh trec06/ham25-delay/
      run.sh trec06/ham50-delay/
      run.sh trec06/spam25-delay/
      run.sh trec06/spam50-delay/

5. Corpus is compatible with "TREC Spam Filter Evaluation Toolkit"
   using the commands:

      run.sh trec06/full/
      run.sh trec06/ham25/
      run.sh trec06/ham50/
      run.sh trec06/spam25/
      run.sh trec06/spam50/

      run.sh trec06/full-delay/
      run.sh trec06/ham25-delay/
      run.sh trec06/ham50-delay/
      run.sh trec06/spam25-delay/
      run.sh trec06/spam50-delay/

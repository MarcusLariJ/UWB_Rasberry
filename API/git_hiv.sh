#!/usr/bin/bash
echo "Lazy git update"
GITUSER="MarcuslariJ"
GITPASS="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"
REPO="https://$GITUSER:$GITPASS@github.com/$GITUSER/UWB_Rasberry.git"
echo "deleting pesky log"
rm log.csv
git fetch "$REPO"
git pull "$REPO"
echo "Done!"
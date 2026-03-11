#include <stdio.h>

#define MAX 100

int processes[MAX];
int n;  // Number of processes
int coordinator = -1;

void holdElection(int initiator) {
    printf("\nProcess %d initiates election.\n", initiator);
    int higherResponded = 0;

    for (int i = initiator + 1; i < n; i++) {
        if (processes[i] == 1) {
            printf("Process %d responds to election message from Process %d.\n", i, initiator);
            higherResponded = 1;
        }
    }

    if (higherResponded) {
        for (int i = initiator + 1; i < n; i++) {
            if (processes[i] == 1) {
                holdElection(i);  // Let the higher process take over the election
                return;
            }
        }
    } else {
        coordinator = initiator;
        printf("Process %d becomes the new coordinator.\n", coordinator);
    }
}

int main() {
    int failed;

    printf("Enter the number of processes: ");
    scanf("%d", &n);

    // Initialize all processes as alive (1)
    for (int i = 0; i < n; i++) {
        processes[i] = 1;
    }

    printf("Enter the ID of the failed coordinator process: ");
    scanf("%d", &failed);

    // Mark the failed process as down
    if (failed >= 0 && failed < n) {
        processes[failed] = 0;
    } else {
        printf("Invalid process ID.\n");
        return 1;
    }

    int initiator;
    printf("Enter the ID of the process initiating the election: ");
    scanf("%d", &initiator);

    if (processes[initiator] == 0) {
        printf("Initiator process is not alive.\n");
        return 1;
    }

    holdElection(initiator);

    printf("\nFinal Coordinator is Process %d\n", coordinator);

    return 0;
}

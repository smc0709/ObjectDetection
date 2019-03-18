/////////////////////////////////////////////////////////////////////////////
//                                                                         //
//     OBJECTS DETECTION AND TRACKING VIA PERCEPTUAL RELEVANCE METRICS     //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////



///  FILE INCLUDES  ///
#include "./PRMovement/pr_movement.h"
#include "./PRMovement/bmpreader.h"
#include "./PRMovement/yuv_rgb.h"
#include "./PRMovement/imgutils.h"
#include "./PRMovement/perceptual_relevance_api.h"
#include <math.h>
//#include <stdio.h>
//#include <stdint.h>
#include <time.h>


///  PREPROCESSOR MACROS  ///
#define BUFF_SIZE_OBJECT_DETECTION 2
#define MAX_NUM_RECTANGLES 10

#define MIN_PR_DIFF_TO_CONSIDER_CUMULUS 0.25    //0.25
#define THRESHOLD_KEEP_RECTANGLE_EDGE 0.25      //0.3

#define TOP_EDGE 1
#define LEFT_EDGE 2
#define RIGHT_EDGE 3
#define BOTTOM_EDGE 4

#define POSSIBLE_CUMULUS 1
#define NOT_A_CUMULUS 0

#define MAX_CUMULUS_SIZE 5

#define MODE_BASE_IMAGE_FIRST_FRAME 1
#define MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME 2

#define MAX_MOV_VALID_FRAME_MODE_FIRST_FRAME 0.065
#define MAX_MOV_VALID_FRAME_MODE_PREV_FRAME 0.035
#define MOV_INCR_PER_FRAME 0.01
#define VALID_FRAME 1
#define INVALID_FRAME 0

#define MAX_RECTANGLE_CHANGE_PER_FRAME 1
#define MAX_FRAMES_NOT_SEEN_UNTIL_REMOVING_RECTANGLE 10 //5 //15 //120      // >= this --> eliminate
#define MIN_FRAMES_SEEN_TO_MAKE_RECTANGLE_PERSISTENT 3 //5                  // >= this --> persist

#define RECT_STATUS_NOT_PERSISTENT 0
#define RECT_STATUS_NORMAL 1
#define RECT_STATUS_DISAPPEARING 2

#define USABLE_BORDER -1    // Number of blocks from the edge inwards of mask with value NULL (where objects are searched).
                            // Value -1 makes all the frame available for searching (mask filled with NULLs).

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define X1_CORNER_FROM_CUMULUS(x_center, cumulus_size) (x_center - (cumulus_size-(1+((cumulus_size-1)%2)))/2)
#define Y1_CORNER_FROM_CUMULUS(y_center, cumulus_size) (y_center - (cumulus_size-(1+((cumulus_size-1)%2)))/2)
#define X2_CORNER_FROM_CUMULUS(x_center, cumulus_size) (x_center + (cumulus_size-(cumulus_size%2))/2)
#define Y2_CORNER_FROM_CUMULUS(y_center, cumulus_size) (y_center + (cumulus_size-(cumulus_size%2))/2)

#define RECT_STATUS(lr_pointer) \
    (((lr_pointer)->frames_seen < MIN_FRAMES_SEEN_TO_MAKE_RECTANGLE_PERSISTENT) ?\
        (RECT_STATUS_NOT_PERSISTENT) :\
        (((lr_pointer)->frames_not_seen == 0) ?\
            (RECT_STATUS_NORMAL) :\
            (RECT_STATUS_DISAPPEARING)\
        )\
    )

#define OUTPUT_COLOURED_FRAME 0     // 1 to output the frame with rectangles, 0 to output black&white blocks
#define MEASURE_TIME_ELAPSED 0      // 1 to measure, 0 not to measure

#define TRACE_LEVEL 2   // How much information is printed in the console. The higher, the more info is printed
                        // -1 shows nothing, 0 shows basic, etc. Accepted values: -1, 0, 1, 2, 3



///  GLOBAL VARIABLES  ///

/**
 * Rectangle: represents the object locations (2 ints define left upper corner and 2 other ints, the right lower corner)
 *  x1, y1
 *  +------------------+
 *  |                  |
 *  |                  |
 *  |                  |
 *  +------------------+ x2, y2
 */
typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
} Rectangle;

/**
 * Cumulus: represents the zones where there are some blocks with "relatively high" pr difference respect to the ref frame
 *   +---------------+  -+           -+
 *   |               |   |            |
 *   |               |   |- size      |
 *   |     (x, y)    |   |            |
 *   |       +       |  _|            |- 2*size - 1
 *   |               |                |
 *   |               |                |
 *   |               |                |
 *   +---------------+               -+
 */
typedef struct {
    int x_center;
    int y_center;
    int cumulus_size;
} Cumulus;

// The number of rectangles
int num_rects;

// The mode used to detect objects (reference frame is always the first o the immediately previous to the current)
int mode;

// Simple linked list (with head and tail pointers) for saving the rectangles
typedef struct linkedrectangle{
    Rectangle *data;
    int frames_not_seen;        // consecutive frames in which the object has not been re-detected
    int frames_seen;            // total number of frames in which the object has been detected
    struct linkedrectangle *next;
} LinkedRectangle;
LinkedRectangle *rect_list_head;
LinkedRectangle *rect_list_tail;

// A mask indicating if the object detector should find rectangles in some blocks or not (rectangles should be able to
// overlap only when moving, not in finding). 0 indicates available for searching otherwise not. 0 --> no rectangle
// 1 --> border around reactangle, 2 --> rectangle itself
int **mask;



///  FUNCTION PROTOTYPES  ///
void rectangles_free();
LinkedRectangle* rectangle_list_add(Rectangle **rect);
int rectangle_list_remove(LinkedRectangle **lr);

void init_mask();
void mask_free();
void fill_mask_zeros();
void compute_mask_for_rect(LinkedRectangle **lr);
void add_rectangle_to_mask(LinkedRectangle **lr);
void print_mask();

void pr_changes(int image_position_buffer);

int find_objects();

int track_objects();
LinkedRectangle* track_object(LinkedRectangle **lr);

int is_cumulus_seed(int block_x, int block_y);

Cumulus get_cumulus_centered(int block_x, int block_y);
float* cumulus_pr_neighbours(int block_x, int block_y, int cumulus_size);
float sum_pr_diffs(int x_center, int y_center, int cumulus_size, int use_mask);
Rectangle* cumulus_to_rectangle(Cumulus cumulus);

void reduce_rectangle_size(Rectangle *rect);
int drop_upper_rows(Rectangle rect, int use_mask);
int drop_lower_rows(Rectangle rect, int use_mask);
int drop_left_columns(Rectangle rect, int use_mask);
int drop_right_columns(Rectangle rect, int use_mask);

int drawEdgeOfRectangle(int block_x, int block_y, int whichEdge, int rect_status);
void draw_rectangles_in_frame();

void update_reference_frame(int position);
int is_frame_valid (int position);



///  FUNCTION IMPLEMENTATIONS  ///

// If the mode that uses first frame as reference is finally deleted, this method can be substituted by pr_to_movement()
// from the pereptual_relevance_api.h library
/**
 * Calculates the differences of the current frame with the reference one, and saves the result in the buffer.
 *
 * @param int image_position_buffer
 *      The position of the of the current image buffer.
 */
void pr_changes(int image_position_buffer) {

    int prev_image_position_buffer;
    switch (mode) {
        case MODE_BASE_IMAGE_FIRST_FRAME:
            for (int i = 0; i < total_blocks_width + 1; i++){
                for (int j = 0; j < total_blocks_height + 1; j++){
                    if (image_position_buffer==0) {
                        diffs_x[j][i] = pr_x_buff[0][j][i];
                        diffs_y[j][i] = pr_y_buff[0][j][i];
                    } else {
                        diffs_x[j][i] = pr_x_buff[image_position_buffer][j][i] - pr_x_buff[0][j][i];
                        diffs_y[j][i] = pr_y_buff[image_position_buffer][j][i] - pr_y_buff[0][j][i];
                    }
                    if (diffs_x[j][i] < 0) diffs_x[j][i] = -diffs_x[j][i];
                    if (diffs_y[j][i] < 0) diffs_y[j][i] = -diffs_y[j][i];
                }
            }
            break;


        case MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME:
            prev_image_position_buffer = image_position_buffer - 1;
            if (prev_image_position_buffer < 0)
                prev_image_position_buffer = buff_size - 1;

            for (int i = 0; i < total_blocks_width + 1; i++){
                for (int j = 0; j < total_blocks_height + 1; j++){
                    diffs_x[j][i] = pr_x_buff[image_position_buffer][j][i] - pr_x_buff[prev_image_position_buffer][j][i];
                    diffs_y[j][i] = pr_y_buff[image_position_buffer][j][i] - pr_y_buff[prev_image_position_buffer][j][i];
                    if (diffs_x[j][i] < 0) diffs_x[j][i] = -diffs_x[j][i];
                    if (diffs_y[j][i] < 0) diffs_y[j][i] = -diffs_y[j][i];
                }
            }
            break;
        default:
            printf("ERROR: mode not defined.\n");
            return;
    }
}

/**
 * Allocates memory for the mask, which dimensions will be total_blocks_height * total_blocks_width. Note: mask not filled.
 */
void init_mask(){
    mask = (int **)malloc(total_blocks_height * sizeof(int *));
    if (mask == NULL) return;      // No memory

    for (int y = 0; y < total_blocks_height; ++y){
        mask[y] = (int *)malloc(total_blocks_width * sizeof(int));
        if (mask[y] == NULL) {          // No memory
            for (int i = y-1; i >=0; --i) {
                free(mask[i]);
                mask[i] = NULL;
            }
            free(mask);
            mask = NULL;
        }
    }
}

/**
 * Frees the memory assigned for the mask and sets it to null.
 */
void mask_free(){
    if (mask==NULL) return;
    for (int y = 0; y < total_blocks_height; ++y){
        free(mask[y]);
        mask[y] = NULL;
    }
    free(mask);
    mask = NULL;
}

/**
 * Computes an overall mask taking into account the mask of all the rectangles in the frame but the one for which is
 *      computed. This way, the mask will show where all the other rectangles are, and therefore make possible to know
 *      where does it can move, where not, and where would be overlapping if it moves.
 *
 * @param LinkedRectangle **lr
 *      The rectangle which needs not to be taken into account for the mask. If this is NULL, mask uses all rectangles.
 */
void compute_mask_for_rect(LinkedRectangle **lr){
    if (mask==NULL){
        init_mask();
        if (mask==NULL) {
            printf("ERROR: no memory\n");
            return;
        }
    }

    fill_mask_zeros();
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return; // all null (list empty). Filled 0s mask
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) return; // at least one null (list corrupt)
    else{
        LinkedRectangle *p = rect_list_head;
        LinkedRectangle *p_before=NULL;
        do {
            if (lr==NULL || p!=*lr) add_rectangle_to_mask(&p);
            p_before = p;
            p = p->next;
        } while (p_before!=rect_list_tail);
    }
}

/**
 * Fills the rectangle mask with zeros everywhere, so that it cleans the memory zone.
 */

void fill_mask_zeros(){

    for (int y = 0; y < total_blocks_height; ++y) {
        for (int x = 0; x < total_blocks_width; ++x) {
            mask[y][x] = 0;
        }
    }
}


/**
 * Adds the area occupied by the rectangle to the mask, so that next rectangles found are not overlapping with it.
 *
 * @param LinkedRectangle **lr
 *      The rectangle to add to the mask.
 */
void add_rectangle_to_mask(LinkedRectangle **lr){
    Rectangle *rect = (*lr)->data;

    for (int x = (rect->x1)-1; x <= (rect->x2)+1; ++x) {        // -1 and +1 in the loop due to extra border
        for (int y = (rect->y1)-1; y <= (rect->y2)+1; ++y) {    // -1 and +1 in the loop due to extra border
            if (!(x<0 || y<0 || x>total_blocks_width-1 || y>total_blocks_height-1)){
                if (x <= (rect->x2) && x >= (rect->x1) && y <= (rect->y2) && y >= (rect->y1)) {
                    mask[y][x] = 2;
                } else if (mask[y][x]==0) {
                    mask[y][x] = 1;
                }
            }
        }
    }
}

/**
 * Prints the mask in the console.
 */
void print_mask() {
    printf("\nMASK:\n");
    for (int y = 0; y < total_blocks_height; ++y) {
        for (int x = 0; x < total_blocks_width; ++x) {
            if (mask[y][x] == 0) {
                printf("- ");
            } else if (mask[y][x] == 1) {
                printf("1 ");
            } else {
                printf("2 ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Frees the resources allocated for storing the rectangles.
 */
void rectangles_free() {
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return; // all null (list empty)
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) printf("ERROR: list corrupt.\n");  // at least one null (list corrupt)
    else{
        LinkedRectangle *p = rect_list_head;
        LinkedRectangle *p_before;
        do {
            p_before = p;
            p = p->next;
            rectangle_list_remove(&p_before);
        } while (p!=NULL);
    }
    if (!(rect_list_head==NULL && rect_list_tail==NULL && num_rects==0)) printf("ERROR while freeing rectangles list.\n");
}

/**
 * Adds the specified rectangle to the end of the linked list. Updates num_rects(+=1), *rect_list_head and *rect_list_tail 
 *      and allocates memory space for the node.
 *
 * @param Rectangle **rect
 *      The pointer to the rectangle to be added to the list. Note: rectangle memory must be already allocated.
 * 
 * @return LinkedRectangle*
 *      The rectangle node created.
 */
LinkedRectangle* rectangle_list_add(Rectangle **rect) {
    if (num_rects>=MAX_NUM_RECTANGLES){                                         // list is full
        free(*rect);
        (*rect) = NULL;
        return NULL;
    }
    
    LinkedRectangle *lr = (LinkedRectangle *)malloc(sizeof(LinkedRectangle));
    if (lr==NULL) {                                                             // no memory
        free(*rect);
        (*rect) = NULL;
        return NULL;
    }

    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0)           // all null (list empty)
        rect_list_head = lr; // the new rectangle is the first element of the list
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0)      // one null but the other(s) not
        return NULL;    // nonsense, both first and last must be null or not
    else                                                                        // all not null (typical non-empty list)
        rect_list_tail->next = lr;

    rect_list_tail = lr; // the new rectangle is the last element of the list
    lr->data = *rect;
    lr->frames_not_seen = 0;
    lr->frames_seen = 1;
    lr->next = NULL;
    num_rects++;

    if (TRACE_LEVEL>=1) printf("Saved rectangle:\tx1=%d\ty1=%d\tx2=%d\ty2=%d\n", (lr->data)->x1, (lr->data)->y1, (lr->data)->x2, (lr->data)->y2);
    return lr;
}

/**
 * Removes a specified rectangle from the linked list.  Updates num_rects(-=1), *rect_list_head and *rect_list_tail and
 *      frees memory space of the node.
 *
 * @param LinkedRectangle **lr
 *      The rectangle to remove from the list.
 * 
 * @return int
 *      Returns 0 if node was removed correctly, 1 in other cases (null reference, empty list, ect.)
 */
int rectangle_list_remove(LinkedRectangle **lr) {
    if (*lr==NULL) return 1;                           // bad parameter
    if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) return 1;      // any null (list empty or corrupt)
    
    if (*lr==rect_list_head) {
        if (*lr==rect_list_tail) {                      // is head and tail
            rect_list_tail = NULL;
            rect_list_head = NULL;
        }else{                                          // is only head
            rect_list_head = rect_list_head->next;
        }
        goto FREES;
    }

    LinkedRectangle *p = rect_list_head;
    LinkedRectangle *p_before = rect_list_head;
    while (p_before!=rect_list_tail) {
        p = p->next;
        if (p==*lr) {
            if (*lr==rect_list_tail) {                  // is only tail
                rect_list_tail = p_before;
                p_before->next = NULL;
            }else {                                     // is typical node
                p_before->next = p->next;
            }
            goto FREES;
        }
        p_before=p;
    }
    return 1;   // rectangle not removed (not in the list!!)

    FREES:
    free((*lr)->data);
    (*lr)->data = NULL;
    free(*lr);
    *lr = NULL;
    num_rects--;
    return 0;
}

/**
 * Computes the sum of the pr differences (with the reference image) of all the blocks in the cumulus. If the given center
 *      is out of the borders of the image, a value of 0.0 is returned, in order not to be selected as the best center.
 *      Out of bounds blocks are not included in the sum.
 *
 * @param int x_center
 *      The x coordinate of the block which is the center of the cumulus.    
 * @param int y_center
 *      The y coordinate of the block which is the center of the cumulus.
 * @param int cumulus_size
 *      Indicates the size of the cumulus. 1 means 1x1 cumulus, 2 means 2x2, etc.
 * @param int use_mask
 *      Indicates if the mask is taken into account or not. 1 to use, 0 to not.
 *
 * @return float
 *      The sum of the pr differences (with the reference image) of all the blocks in the cumulus.
 */
float sum_pr_diffs(int x_center, int y_center, int cumulus_size, int use_mask) {
    if ((x_center<0 || y_center<0 || x_center>total_blocks_width || y_center>total_blocks_height)) {
        return 0.0;
    }

    float sum;
    int x_min, y_min, x_max, y_max;

    x_min = X1_CORNER_FROM_CUMULUS(x_center, cumulus_size);
    y_min = Y1_CORNER_FROM_CUMULUS(y_center, cumulus_size);
    x_max = X2_CORNER_FROM_CUMULUS(x_center, cumulus_size);
    y_max = Y2_CORNER_FROM_CUMULUS(y_center, cumulus_size);

    sum = 0.0;
    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            if (!(x<0 || y<0 || x>total_blocks_width-1 || y>total_blocks_height-1)){       // if NOT out of image bounds
                if (!use_mask || (use_mask && mask[y][x] == 0)) {
                    sum += get_block_movement(x, y);
                } else{
                    return 0.0;     // no sum is done when overlapping. A value of 0.0 is returned
                }
            }
        }
    }
    return sum;
}

/**
 * Calculates the pr sums of all blocks of each possible cumulus.  The possible centers (8 closest neighbours and the
 *      current one). To calculate the pr sum of a cumulus, takes into account all the blocks inside the cumulus_size
 *      radius.
 * 
 * @param int block_x
 *      The x block coordinate.
 * @param int block_y
 *      The y block coordinate.
 * @param int cumulus_size
 *      Indicates the size of the cumulus. 1 means 1x1 cumulus, 2 means 2x2, etc.
 *
 * @return float*
 *      Returns the array of sums of PRs of each possible center. Always 9 positions with allocated memory (remember to
 *      free it).
 */
float* cumulus_pr_neighbours(int block_x, int block_y, int cumulus_size) {
    int ij_index;
    float* pr_values = malloc(9 * sizeof(float));
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
           ij_index = (j+1) + 3*(i+1);
           pr_values[ij_index] = sum_pr_diffs(block_x+j, block_y+i, cumulus_size, 1);
        }
    }
    return pr_values;
}

/**
 * Checks if a block and its closest environment is relevant enough to be a cumulus.
 *
 * @param int block_x
 *      The x coordinate of the block that is asked to be considered cumulus.    
 * @param int block_y
 *      The y coordinate of the block that is asked to be considered cumulus.
 *
 * @return int
 *      Returns NOT_A_CUMULUS if the coordinates do not meet the requirements to be considered a cumulus, and returns
 *      POSSIBLE_CUMULUS if they are met.
 */
int is_cumulus_seed(int block_x, int block_y) {
    float *pr_values;
    int is_cumulus = NOT_A_CUMULUS;

    if (sum_pr_diffs(block_x, block_y, 1, 1) >= MIN_PR_DIFF_TO_CONSIDER_CUMULUS) {
        pr_values = cumulus_pr_neighbours(block_x, block_y, 1);

        if (!(pr_values[0]==0.0 && pr_values[1]==0.0 && pr_values[2]==0.0 && \
              pr_values[3]==0.0 &&                      pr_values[5]==0.0 && \
              pr_values[6]==0.0 && pr_values[7]==0.0 && pr_values[8]==0.0)) {
            is_cumulus = POSSIBLE_CUMULUS;
        }
        free(pr_values);
        pr_values = NULL;
    }
    return is_cumulus;
}

/**
 * Given a block inside a cumulus, finds the Cumulus with the best-fit block as center.
 *
 * @param int block_x
 *      The x coordinate of the starting center.    
 * @param int block_y
 *      The y coordinate of the starting center.
 *
 * @return Cumulus
 *      The cumulus with the selected center and size.
 */
Cumulus get_cumulus_centered(int block_x, int block_y) {
    int cumulus_size, index_new_center;
    float *pr_values;
    Cumulus cumulus;
    cumulus.x_center = block_x;
    cumulus.y_center = block_y;

    for (cumulus_size = 3; cumulus_size <= MAX_CUMULUS_SIZE; ++cumulus_size) {
        do{
            index_new_center = 4;   // The current center. If we dont move, we will stop iterating
            pr_values = cumulus_pr_neighbours(cumulus.x_center, cumulus.y_center, cumulus_size);
            for (int i = 0; i < 9; ++i) {
                if (pr_values[index_new_center] < pr_values[i]) {
                    index_new_center = i;
                }
                if (TRACE_LEVEL>=3) printf("pr_values[%d] = %f\n", i, pr_values[i]);
            }
            // Update center coordinates
            if (index_new_center != 4) {
                cumulus.x_center += (index_new_center%3)-1;
                cumulus.y_center += (index_new_center/3)-1;
            }
            free(pr_values);
            pr_values = NULL;
        } while (index_new_center!=4);
        cumulus.cumulus_size = cumulus_size;
    }
    // Do not let the cumulus to be out of frame. Move it inside
    while (X1_CORNER_FROM_CUMULUS(cumulus.x_center, cumulus_size) < 0)                      // (x1 < 0)
        cumulus.x_center++;
    while (Y1_CORNER_FROM_CUMULUS(cumulus.y_center, cumulus_size) < 0)                      // (y1 < 0)
        cumulus.y_center++;
    while (X2_CORNER_FROM_CUMULUS(cumulus.x_center, cumulus_size) > total_blocks_width)     // (x2 > width)
        cumulus.x_center--;
    while (Y2_CORNER_FROM_CUMULUS(cumulus.y_center, cumulus_size) > total_blocks_height)    // (y2 > height)
        cumulus.y_center--;
    return cumulus;
}

/**
 * Transforms a cumulus to a rectangle. Allocates memory for the rectangle.
 *
 * @param Cumulus cumulus
 *      The cumulus to transform.    
 *
 * @return *Rectangle
 *      The rectangle that encloses the cumulus. Remember to free memory.
 */
Rectangle* cumulus_to_rectangle(Cumulus cumulus){
    Rectangle *rect = (Rectangle *)malloc(sizeof(Rectangle));
    if (rect==NULL){
        printf("ERROR: no memory\n");
        return NULL;
    }

    rect->x1 = X1_CORNER_FROM_CUMULUS(cumulus.x_center, cumulus.cumulus_size);
    rect->y1 = Y1_CORNER_FROM_CUMULUS(cumulus.y_center, cumulus.cumulus_size);
    rect->x2 = X2_CORNER_FROM_CUMULUS(cumulus.x_center, cumulus.cumulus_size);
    rect->y2 = Y2_CORNER_FROM_CUMULUS(cumulus.y_center, cumulus.cumulus_size);

    return rect;
}

/**
 * These four functions ("drop_side_row/col()") keep the same structure. They find the number of border lines that can be
 *      eliminated from one side (top/bottom/left/right) of the rectangle to keep it the smallest but still emcompassing
 *      the object.
 *
 * @param Rectangle rect
 *      The rectangle to work with.
 * @param int use_mask
 *      Indicates if the mask is taken into account or not. 1 to use, 0 to not.
 *
 * @return int
 *      The number of lines to remove (0 to keep it the same).
 */
int drop_upper_rows(Rectangle rect, int use_mask) {
    float max_pr_in_the_line;
    int deleted_rows = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2; 

    for (int i = y1; i <= y2; ++i) {
        max_pr_in_the_line = 0.0;
        for (int j = x1; j <=x2; ++j) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1, use_mask));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_rows++;
        } else {
            return deleted_rows;
        }
    }
    return deleted_rows;
}
int drop_lower_rows(Rectangle rect, int use_mask) {
    float max_pr_in_the_line;
    int deleted_rows = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2;

    for (int i = y2; i >= y1; --i) {
        max_pr_in_the_line = 0.0;
        for (int j = x1; j <= x2; ++j) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1, use_mask));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_rows++;
        } else {
            return deleted_rows;
        }
    }
    return deleted_rows;
}
int drop_left_columns(Rectangle rect, int use_mask) {
    float max_pr_in_the_line;
    int deleted_cols = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2;

    for (int j = x1; j <= x2; ++j) {
        max_pr_in_the_line = 0.0;
        for (int i = y1; i <= y2; ++i) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1, use_mask));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_cols++;
        } else {
            return deleted_cols;
        }
    }
    return deleted_cols;
}
int drop_right_columns(Rectangle rect, int use_mask) {
    float max_pr_in_the_line;
    int deleted_cols = 0;
    int x1 = rect.x1;
    int y1 = rect.y1;
    int x2 = rect.x2;
    int y2 = rect.y2;

    for (int j = x2; j >= x1; --j) {
        max_pr_in_the_line = 0.0;
        for (int i = y1; i <= y2; ++i) {
            max_pr_in_the_line = MAX(max_pr_in_the_line, sum_pr_diffs(j, i, 1, use_mask));
        }
        if (max_pr_in_the_line < THRESHOLD_KEEP_RECTANGLE_EDGE) {
            deleted_cols++;
        } else {
            return deleted_cols;
        }
    }
    return deleted_cols;
}

/**
 * Given a rectangle, keeps or reduces its size to convert it in the smallest rectangle possible that encloses the object.
 *
 * @param *Rectnagle rect
 *      The rectangle to work with.    
 */
void reduce_rectangle_size(Rectangle *rect) {
    (rect->y1) += drop_upper_rows(*rect, 1);
    (rect->y2) -= drop_lower_rows(*rect, 1);
    (rect->x1) += drop_left_columns(*rect, 1);
    (rect->x2) -= drop_right_columns(*rect, 1);
}

/**
 * Finds the objects in the frame. Starts analyzing the possible cumuli of blocks with high pr difference relative to the
 *      base image. After an iterative process of finding the best cumuli center and refining the size, defines the
 *      rectangle that encloses the object and saves it.
 *
 * @return int
 *      0 if everything went fine. 1 if the maximum number of rectangles was reached.
 */
int find_objects() {
    Rectangle *rect;
    Cumulus cumulus;
    LinkedRectangle *lr;
    compute_mask_for_rect(NULL);
    if (TRACE_LEVEL>=2) print_mask();
    if (TRACE_LEVEL>=1) printf("\nSearching...\n");
    for (int block_y=0; block_y<total_blocks_height; block_y++) {
        for (int block_x=0; block_x<total_blocks_width; block_x++) {
            if (TRACE_LEVEL>=3) printf("block_x=%d - - - block_y=%d\n", block_x, block_y);

            if (is_cumulus_seed(block_x, block_y)){
                if (TRACE_LEVEL>=1) printf("Cumulus seed found: \tblock_x = %d \tblock_y = %d\n", block_x, block_y);
                cumulus = get_cumulus_centered(block_x, block_y);

                rect = cumulus_to_rectangle(cumulus);
                reduce_rectangle_size(rect);

                lr = rectangle_list_add(&rect);
                compute_mask_for_rect(NULL);

                if (num_rects >= MAX_NUM_RECTANGLES){
                    if (TRACE_LEVEL>=1) printf("Maximum number of rectangles reached\n");
                    return 1;
                }
            }
        }
    }
    if (TRACE_LEVEL>=2) print_mask();
    return 0;
}

/**
 * Tracks the objects in the previous frame, updating their position and size.
 *
 * @return int
 *      Returns 0 if everything was OK, other values if something went wrong. Note: -1 if list is corrupt.
 */
int track_objects(){
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return 0; // all null (list empty)
    else if (rect_list_head==NULL || rect_list_tail==NULL || num_rects==0) return -1; // at least one null (list corrupt)
    else{
        LinkedRectangle *p = rect_list_head;
        LinkedRectangle *p_before;
        do {
            p_before = p;
            p = track_object(&p);
            if (p == NULL) return 0;            // si se borra el último de la lista
        } while (p_before!=rect_list_tail);
    }
    return 0;
}

/**
 * Updates the rectangle position and size. There is a max size grow and a max movement.
 *
 * @param LinkedRectangle **lr
 *      The rectangle to be tracked (keep, update position and size, or delete).
 * 
 * @return LinkedRectangle *next
 *      The next rectangle in the list.
 */
LinkedRectangle* track_object(LinkedRectangle **lr){
    int initial_x1, initial_x2, initial_y1, initial_y2;
    int upper_rows, lower_rows, left_columns, right_columns;
    int x1_enlarged, x2_enlarged, y1_enlarged, y2_enlarged, x1_enlargement, x2_enlargement, y1_enlargement, y2_enlargement;
    int use_mask;

    if ((*lr)==NULL) return NULL;

    Rectangle *rect = (*lr)->data;
    LinkedRectangle *next = (*lr)->next;
    int frames_not_seen = (*lr)->frames_not_seen;
    int frames_seen = (*lr)->frames_seen;

    if (RECT_STATUS(*lr) == RECT_STATUS_NOT_PERSISTENT)
        use_mask = 1;
    else
        use_mask = 0;

    compute_mask_for_rect(NULL);
     
    // Save initial values
    initial_x1 = rect->x1;
    initial_x2 = rect->x2;
    initial_y1 = rect->y1;
    initial_y2 = rect->y2;

    // Enlage rectangle
    //rect->x1 = MAX(rect->x1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    //rect->x2 = MIN(rect->x2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_width);
    //rect->y1 = MAX(rect->y1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    //rect->y2 = MIN(rect->y2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_height);

    x1_enlarged = MAX(rect->x1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    x1_enlargement = rect->x1 - x1_enlarged;
    rect->x1 = x1_enlarged;

    x2_enlarged = MIN(rect->x2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_width-1);
    x2_enlargement = rect->x2 - x2_enlarged;
    rect->x2 = x2_enlarged;

    y1_enlarged = MAX(rect->y1 - MAX_RECTANGLE_CHANGE_PER_FRAME, 0);
    y1_enlargement = rect->y1 - y1_enlarged;
    rect->y1 = y1_enlarged;

    y2_enlarged = MIN(rect->y2 + MAX_RECTANGLE_CHANGE_PER_FRAME, total_blocks_height-1);
    y2_enlargement = rect->y2 - y2_enlarged;
    rect->y2 = y2_enlarged;


    upper_rows    = drop_upper_rows(*rect, use_mask);
    rect->y1 += MIN(upper_rows, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);

    lower_rows    = drop_lower_rows(*rect, use_mask);
    rect->y2 -= MIN(lower_rows, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);

    left_columns  = drop_left_columns(*rect, use_mask);
    rect->x1 += MIN(left_columns, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);

    right_columns = drop_right_columns(*rect, use_mask);
    rect->x2 -= MIN(right_columns, 2*MAX_RECTANGLE_CHANGE_PER_FRAME);
    
    if (TRACE_LEVEL>=0) printf("\n---RECTANGLE---\n");
    if (TRACE_LEVEL>=2) printf("INICIAL  -->\tx1=%d\t y1=%d\t x2=%d\t y2=%d\n", initial_x1, initial_y1, initial_x2, initial_y2);
    if (TRACE_LEVEL>=2) printf("ENLARGED -->\tx1=%d\t y1=%d\t x2=%d\t y2=%d\n", x1_enlarged, y1_enlarged, x2_enlarged, y2_enlarged);
    if (TRACE_LEVEL>=2) printf("FINAL    -->\t");
    if (TRACE_LEVEL>=0)               printf("x1=%d\t y1=%d\t x2=%d\t y2=%d\n", rect->x1, rect->y1, rect->x2, rect->y2);
    if (TRACE_LEVEL>=0) printf("frames_not_seen = %d\n", frames_not_seen);
    if (TRACE_LEVEL>=0) printf("frames_seen = %d\n", frames_seen);

    // If the rectange needs to be smaller than Nx1 or 1xN, we assume it dissappeared, and keep it the same but modifying a
    // counter until some grace frames confirm that the object dissapeared or stopped moving. Also if the rectangle is not
    // persistent and needs to decrease (vertically or horizontally) more than twice the MAX_RECTANGLE_CHANGE_PER_FRAME, it
    // is removed.
    printf("upper_rows=%d, lower_rows=%d, left_columns=%d, right_columns=%d\n", upper_rows, lower_rows, left_columns, right_columns);
    if (upper_rows + lower_rows >= 1 + y2_enlarged - y1_enlarged  ||  left_columns + right_columns >= 1 + x2_enlarged - x1_enlarged \
        ||  (use_mask==1 && (upper_rows + lower_rows > 2*2*MAX_RECTANGLE_CHANGE_PER_FRAME || left_columns + right_columns > 2*2*MAX_RECTANGLE_CHANGE_PER_FRAME))) {
        rect->y1 = initial_y1;
        rect->y2 = initial_y2;
        rect->x1 = initial_x1;
        rect->x2 = initial_x2;
        frames_not_seen++;
        if (frames_not_seen >= MAX_FRAMES_NOT_SEEN_UNTIL_REMOVING_RECTANGLE \
            || frames_seen < MIN_FRAMES_SEEN_TO_MAKE_RECTANGLE_PERSISTENT) {
            rectangle_list_remove(lr);
            if (TRACE_LEVEL>=1) printf("Rectangle removed.\n");
            *lr = NULL;
            return next;
        }
    } else{
        frames_seen++;
        frames_not_seen = 0;
    }
    (*lr)->frames_not_seen = frames_not_seen;
    (*lr)->frames_seen = frames_seen;

    return next;
}

/**
 * Draws the specified edge in the block specified changing the luminance and chrominances in the corresponding edge of the
 *      block.
 *
 * @param int block_x
 *      The x coordinate of the block in which the rectangle edge should be drawn.    
 * @param int block_y
 *      The y coordinate of the block in which the rectangle edge should be drawn.
 * @param int whichEdge
 *      Indicates which of the edges should be drawn in the block (TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE)
 * @param int rect_status
 *      Indicates the status of the rectangle which side should be drawn. RECT_STATUS_NOT_PERSISTENT, RECT_STATUS_NORMAL,
 *      RECT_STATUS_DISAPPEARING
 *
 * @return int
 *      If everything is fine returns 0, if there was a problem, -1.
 */
int drawEdgeOfRectangle(int block_x, int block_y, int whichEdge, int rect_status) {
    int y_value, u_value, v_value;
    int xini = block_x*theoretical_block_width;
    int xfin = xini+theoretical_block_width;
    int yini = block_y*theoretical_block_height;
    int yfin = yini+theoretical_block_height;

    if (xfin > width-theoretical_block_width)
        xfin = width;
    if (yfin > height-theoretical_block_height)
        yfin = height;

    if (rect_status == RECT_STATUS_NOT_PERSISTENT) {        // yellow
        y_value = 255;
        u_value = 0;
        v_value = 148;
    } else if (rect_status == RECT_STATUS_NORMAL) {         // red
        y_value = 76;
        u_value = 84;
        v_value = 255;
    } else if (rect_status == RECT_STATUS_DISAPPEARING) {   // pink
        y_value = 135;
        u_value = 195;
        v_value = 213;
    } else{
        printf("ERROR: incorrect rectangle status.\n");
        return -1;
    }

    switch (whichEdge) {
        case TOP_EDGE:
            for (int xx = xini; xx < xfin; xx++) {
                y[yini*width+xx] = y_value;
                y[(yini+1)*width+xx] = y_value;
            }
            for (int xx = xini/2; xx < xfin/2; xx++) {
                u[yini/2*width+xx] = u_value;
                v[yini/2*width+xx] = v_value;
            }
        break;

        case LEFT_EDGE:
            for (int yy = yini; yy < yfin; yy++) {
                y[yy*width+xini] = y_value;
                y[yy*width+(xini+1)] = y_value;
            }
            for (int yy = yini/2; yy < yfin/2; yy++) {
                u[yy*width+xini/2] = u_value;
                v[yy*width+xini/2] = v_value;
            }
        break;

        case RIGHT_EDGE:
            for (int yy = yini; yy < yfin; yy++) {
                y[yy*width+(xfin-1)] = y_value;
                y[yy*width+(xfin-2)] = y_value;
            }
            for (int yy = yini/2; yy < yfin/2; yy++) {
                u[yy*width+(xfin/2-1)] = u_value;
                v[yy*width+(xfin/2-1)] = v_value;
            }       
        break;

        case BOTTOM_EDGE:
            for (int xx = xini; xx < xfin; xx++) {
                y[(yfin-1)*width+xx] = y_value;
                y[(yfin-2)*width+xx] = y_value;
            }
            for (int xx = xini/2; xx < xfin/2; xx++) {
                u[(yfin/2-1)*width+xx] = u_value;
                v[(yfin/2-1)*width+xx] = v_value;
            }
        break;

        default:
            printf("ERROR: incorrect rectangle side.\n");
            return -1;
    }
    return 0;
}

/**
 * Draws the rectangles in the frame changing the luminance and chrominances in the edges of the rectangles.
 */
void draw_rectangles_in_frame() {
    Rectangle *rect;
    LinkedRectangle *p = rect_list_head;
    // for each rectangle
    if (rect_list_head==NULL && rect_list_tail==NULL && num_rects==0) return; // all null (list empty)
    else while (p!=NULL){
        int rect_status;
        if (p->frames_seen < MIN_FRAMES_SEEN_TO_MAKE_RECTANGLE_PERSISTENT)
            rect_status = RECT_STATUS_NOT_PERSISTENT;   // Non-confirmed object rectangle (initial)
        else if (p->frames_not_seen == 0)
            rect_status = RECT_STATUS_NORMAL;           // Normal rectangle (already considered object)
        else
            rect_status = RECT_STATUS_DISAPPEARING;     // Disappearing rectangle (object not seen in last frame(s))

        rect = p->data;
        for (int block_y = rect->y1; block_y <= rect->y2; block_y++) {
            for (int block_x = rect->x1; block_x <= rect->x2; block_x++) {
                // Draw top edges
                if (block_y==rect->y1){
                    drawEdgeOfRectangle(block_x, block_y, TOP_EDGE, rect_status);
                    if (TRACE_LEVEL>=3) printf("TOP_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
                // Draw left edge
                if (block_x==rect->x1){
                    drawEdgeOfRectangle(block_x, block_y, LEFT_EDGE, rect_status);
                    if (TRACE_LEVEL>=3) printf("LEFT_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
                // Draw right edge
                if (block_x==rect->x2){
                    drawEdgeOfRectangle(block_x, block_y, RIGHT_EDGE, rect_status);
                    if (TRACE_LEVEL>=3) printf("RIGHT_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
                // Draw bottom edge
                if (block_y==rect->y2){
                    drawEdgeOfRectangle(block_x, block_y, BOTTOM_EDGE, rect_status);
                    if (TRACE_LEVEL>=3) printf("BOTTOM_EDGE\tblock_x=%d\tblock_y=%d\n", block_x, block_y);
                }
            }
        }
        p = p->next;
    }
}

/**
 * Updates the frame used as reference to the current frame.
 *
 * @param int position
 *      The current buff position.
 */
void update_reference_frame(int position){
    //pr_x_buff[0] = pr_x_buff[position];
    //pr_y_buff[0] = pr_y_buff[position];
    for (int block_y=0; block_y<total_blocks_height+1; block_y++) {
        for (int block_x=0; block_x<total_blocks_width+1; block_x++) {
            pr_x_buff[0][block_y][block_x] = pr_x_buff[position][block_y][block_x];
            pr_y_buff[0][block_y][block_x] = pr_y_buff[position][block_y][block_x];
        }
    }
}

/**
 * Verifies if a frame is valid or not.
 *
 * @param int block_x
 *      The x coordinate of the block in which the rectangle edge should be drawn.    
 * @param int block_y
 *      The y coordinate of the block in which the rectangle edge should be drawn.
 * @param int whichEdge
 *      Indicates which of the edges should be drawn in the block (top, bottom, left or right).
 *
 * @return int
 *      If frame is valid returns VALID_FRAME (1), else returns INVALID_FRAME (0).
 */
int is_frame_valid (int position){
    int is_valid = INVALID_FRAME;
    static int in_a_row_invalid_frames = 0; //first time is 0, then saves value between calls
    static float last_movement = 0.0; //first time is 0.0, then saves value between calls
    float movement = get_image_movement(0);
    //if (movement==0.0) return INVALID_FRAME;

    switch (mode){
        case MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME:
            if (movement < last_movement+MOV_INCR_PER_FRAME || movement <= MAX_MOV_VALID_FRAME_MODE_PREV_FRAME) {
                is_valid = VALID_FRAME;
                last_movement = movement;
                if (TRACE_LEVEL>=0) printf("Movement = %f\n", movement);
            }else{
                if (TRACE_LEVEL>=0) printf("! ! ! --> INVALID FRAME, Movement = %f\n", movement);
            }
            break;

        case MODE_BASE_IMAGE_FIRST_FRAME:
            if (movement <= MAX_MOV_VALID_FRAME_MODE_FIRST_FRAME) {
                is_valid = VALID_FRAME;
                in_a_row_invalid_frames = 0;
                if (TRACE_LEVEL>=0) printf("Movement = %f\n", movement);
            } else {
                if (TRACE_LEVEL>=0) printf("! ! ! --> INVALID FRAME \tMovement = %f\n", movement);
                in_a_row_invalid_frames++;
                if (in_a_row_invalid_frames >= 2) {
                    update_reference_frame(position);
                }
            }
            break;

        default: 
            printf("ERROR: incorrect mode.\n");
            return INVALID_FRAME;
    }

    return is_valid;
}



///  MAIN  ///
int main( int argc, char** argv ) {
    mode = MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME; //MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME   MODE_BASE_IMAGE_FIRST_FRAME
    
    struct timespec time_start, time_end;
    uint64_t time_elapsed = 0;

    if (TRACE_LEVEL>=0) printf("\n\
         ___________________________________________________________________________ \n\
        |                                                                           |\n\
        |      OBJECTS DETECTION AND TRACKING VIA PERCEPTUAL RELEVANCE METRICS      |\n\
        |___________________________________________________________________________|\n");

    char* imageName = argv[1];
    char* imageExt = argv[2];
    char* frameArg = argv[3];
    int frameNumber;
    char image[100];
    int frame = atoi(frameArg);
    initiated = 0;
    mask=NULL;

    int starting_frame;

    if (mode==MODE_BASE_IMAGE_FIRST_FRAME){
        if (TRACE_LEVEL>=0) printf("\nUsing first frame as reference (auto-modified when 2 invalid frames in a row).\n");
        starting_frame = 0;
        buff_size = 2;
    }
    else if (mode == MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME){
        if (TRACE_LEVEL>=0) printf("\nUsing previous frame as reference.\n");
        starting_frame = 1;
        buff_size = BUFF_SIZE_OBJECT_DETECTION;
    }

    // Loop to process all the frames required
    for (frameNumber = starting_frame; frameNumber < frame; frameNumber++){

        if (frameArg == NULL){
            frameNumber = 0;
            strcpy(image, imageName);
            strcat(image, imageExt);
        } else {
            sprintf(frameArg,"%i",frameNumber);
            strcpy(image, imageName);
            strcat(image, frameArg);
            strcat(image, imageExt);
        }

        if (TRACE_LEVEL>=0) printf("\n\n\nInput: %s\n", image);

        BITMAPINFOHEADER bitmapInfoHeader;

        LoadBitmapFileProperties(image, &bitmapInfoHeader);
        width = bitmapInfoHeader.biWidth;
        height = bitmapInfoHeader.biHeight;
        rgb_channels = bitmapInfoHeader.biBitCount/8;
        
        if (TRACE_LEVEL>=3) printf("width = %d, height = %d, rgb_channels = %d\n", width, height, rgb_channels);
        if (initiated == 0) {
            init_pr_computation(width, height, rgb_channels);
            initiated = 1;
        }
        
        rgb = load_frame(image, width, height, rgb_channels);

        const size_t y_stride = width + (16-width%16)%16;
        const size_t uv_stride = y_stride;
        const size_t rgb_stride = width*3 +(16-(3*width)%16)%16;

        rgb24_yuv420_std(width, height, rgb, rgb_stride, y, u, v, y_stride, uv_stride, YCBCR_601);

        free(rgb);
        rgb = NULL;
        
        int position; //int position = MIN(frameNumber, 1); //int position = (frameNumber-1)%BUFF_SIZE_OBJECT_DETECTION;
        if (mode==MODE_BASE_IMAGE_FIRST_FRAME)
            position = MIN(frameNumber, 1);
        else if (mode==MODE_BASE_IMAGE_INMEDIATE_PREVIOUS_FRAME)
            position = (frameNumber-1)%BUFF_SIZE_OBJECT_DETECTION;

        if (MEASURE_TIME_ELAPSED){
            clock_gettime(CLOCK_MONOTONIC, &time_start);
        }

        lhe_advanced_compute_perceptual_relevance (y, pr_x_buff[position], pr_y_buff[position]);

        pr_changes(position);
        
        if (!OUTPUT_COLOURED_FRAME) {
            create_frame(0);
        }

        if (frameNumber > starting_frame) {
            if (is_frame_valid(position)){
                track_objects();
                find_objects();
            }
            draw_rectangles_in_frame();
        }

        if (MEASURE_TIME_ELAPSED){
            clock_gettime(CLOCK_MONOTONIC, &time_end);
            time_elapsed += ((time_end.tv_sec * 1000000000) + time_end.tv_nsec) - ((time_start.tv_sec * 1000000000) + time_start.tv_nsec);
        }

        char frameName[100];
        sprintf(frameName,"./output/output%i.bmp",frameNumber);
        if (TRACE_LEVEL>=1) printf("Output: %s\n", frameName);
        yuv420_rgb24_std(width, height, y, u, v, y_stride, uv_stride, rec_rgb, rgb_stride, YCBCR_601);
        stbi_write_bmp(frameName, width, height, rgb_channels, rec_rgb);

    }
    if (MEASURE_TIME_ELAPSED) {
        printf("\n\nThe total time elapsed is: %lf miliseconds approximately.\n", ((double)time_elapsed/1000000));
        printf("The elapsed time per frame is: %lf miliseconds approximately.\n\n", ((double)time_elapsed/1000000)/(double)frame);
    }
    mask_free();
    close_pr_computation();
    rectangles_free();

    return 0;
}



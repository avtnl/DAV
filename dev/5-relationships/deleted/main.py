    # Initialize variables needed for STEPs 1, 4, 5, 6, 7, and 8
    if 1 in Script or 4 in Script or 5 in Script or 6 in Script or 7 in Script or 8 in Script:
        df, group_authors, non_anthony_group, anthony_group, sorted_groups = data_preparation.build_visual_categories(df)
        if df is None or group_authors is None or sorted_groups is None:
            logger.error("Failed to initialize required variables for STEPs 1, 4, 5, 6, 7, or 8.")
            return

    # STEP 1B: Attachment categories visualization
    if 1 in Script:
        # Create attachment bar chart (overall)
        fig2 = plot_manager.build_visual_categories_2(df)
        if fig2 is None:
            return
        
        # Save attachment bar chart
        png_file2 = file_manager.save_png(fig2, image_dir, filename="attachment_bar_chart")
        if png_file2 is None:
            return
        
    # STEP 4: Relationships visualization per group
    if 4 in Script:
        for group in sorted_groups:
            df_group = df[df['whatsapp_group'] == group].copy()
            if df_group.empty:
                logger.error(f"No data found for WhatsApp group '{group}'. Skipping relationships visualization.")
                continue
            
            table2 = data_preparation.build_visual_relationships(df_group, group_authors[group])
            if table2 is not None:
                fig_rel = plot_manager.build_visual_relationships(table2, group)
                if fig_rel is not None:
                    png_file_rel = file_manager.save_png(fig_rel, image_dir, filename=f"relationships_{group}")
                    if png_file_rel is None:
                        logger.error("Failed to save relationships plot.")

    # STEP 5: Relationships visualization for emoji sequences per group
    if 5 in Script:
        for group in sorted_groups:
            df_group = df[df['whatsapp_group'] == group].copy()
            if df_group.empty:
                logger.error(f"No data found for WhatsApp group '{group}'. Skipping relationships_2 visualization.")
                continue
            
            table1, table2 = data_preparation.build_visual_relationships_2(df_group, group_authors[group])
            if table1 is not None and not table1.empty:
                file_manager.save_table(table1, tables_dir, f"emoji_seq_total_{group}")
            if table2 is not None and not table2.empty:
                file_manager.save_table(table2, tables_dir, f"emoji_seq_highest_{group}")
                fig_rel = plot_manager.build_visual_relationships_2(table2, group)
                if fig_rel is not None:
                    png_file_rel = file_manager.save_png(fig_rel, image_dir, filename=f"relationships_emoji_{group}")
                    if png_file_rel is None:
                        logger.error("Failed to save relationships_2 plot.")